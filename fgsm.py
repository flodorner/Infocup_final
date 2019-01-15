import torch
from torch import optim, nn
from os import remove
import numpy as np
from glob import glob
from time import sleep

from config import *
from utilities import torch_to_array, url_to_torch, save_and_query, query_to_labels, query_names
from whitebox import create_whitebox


def project_l_inf(x, base, bound):
    perturbation = x - base
    perturbation = np.maximum(perturbation, np.zeros(perturbation.shape) - bound)
    perturbation = np.minimum(perturbation, np.zeros(perturbation.shape) + bound)
    return base + perturbation


def project_l_2(x, base, bound):
    perturbation = x - base
    norm = np.linalg.norm(perturbation)
    perturbation = bound * perturbation / max([1, norm])
    return base + perturbation


# noinspection PyCallingNonCallable
class FGSM:
    def __init__(self, model=None, cuda=False):

        if torch.cuda.is_available() and cuda is True:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        if model is None:
            self.model = create_whitebox(self.device)
        else:
            self.model = model

        self.mode = FGSM_SPECS["mode"]
        self.bound = FGSM_SPECS["bound"]
        self.magnitude = FGSM_SPECS["magnitude"]
        self.max_fgsm_iterations = FGSM_SPECS["max_fgsm_iterations"]
        self.target_threshold = FGSM_SPECS["target_threshold"]

        self.fgsm_restart = FGSM_SPECS["fgsm_restart"]
        self.restart_max_amount = FGSM_SPECS["restart_max_amount"]
        self.restart_accuracy_bound = FGSM_SPECS["restart_accuracy_bound"]

        self.retrain_mode = FGSM_SPECS["retrain_mode"]
        self.retrain_lr = FGSM_SPECS["retrain_lr"]
        self.retrain_max_gradient_steps = FGSM_SPECS["retrain_max_gradient_steps"]
        self.retrain_threshold = FGSM_SPECS["retrain_threshold"]
        self.always_save = FGSM_SPECS["always_save"]

        self.print = FGSM_SPECS["print"]

    def _get_gradient(self, im, target_label):
        self.model.eval()
        im_tensor = torch.tensor(im).to(self.device)
        im_tensor.requires_grad = True
        output = self.model(im_tensor)
        loss = torch.take(output, torch.tensor(target_label).to(self.device))
        loss.backward()
        im_grad = im_tensor.grad

        return im_grad

    def _train_on_label(self, images, labels, target_label):
        optimizer = optim.Adam(self.model.parameters(), lr=self.retrain_lr)
        label_num = len(labels[0])
        labels = labels[:, target_label]
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels).float()

        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        images, labels = images.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        output = self.model(images)
        indexes = [i * label_num + target_label for i in range(len(labels))]
        loss = nn.MSELoss()(torch.take(output, torch.tensor(indexes).to(self.device)), labels)
        loss.backward()
        optimizer.step()

        self.model.eval()
        loss = nn.MSELoss()(torch.take(output, torch.tensor(indexes).to(self.device)), labels)

        loss_number = loss.item()

        return loss_number

    def _create_advers(self, im, target_label, base):
        prob = 0
        steps = 0
        while prob < self.target_threshold and steps < self.max_fgsm_iterations:
            steps += 1
            im, prob = self.fastgrad_step(im, target_label, base)
        if self.print:
            print("probability: Whitebox")
            print(prob)
        return im

    def retrain(self, im, label, target_label):
        i = 0
        error = 1
        while error > self.retrain_threshold and i < self.retrain_max_gradient_steps:
            i = i + 1
            error = self._train_on_label(im, label, target_label)
        if self.print:
            print("MSE White vs Black Box after retraining:")
            print(error)
        return i, error

    def get_label(self, im, target_label):
        self.model.eval()
        output = self.model(torch.tensor(im).to(self.device))
        label = torch.take(output, torch.tensor(target_label).to(self.device)).item()
        return label

    def fastgrad_step(self, im, target_label, base):
        im_grad = self._get_gradient(im, target_label)
        im = im + self.magnitude * np.sign(im_grad.cpu().detach().numpy())
        if self.mode == "simple":
            pass
        elif self.mode == "l_2":
            im = project_l_2(im, base, self.bound)
        elif self.mode == "l_inf":
            im = project_l_inf(im, base, self.bound)
        else:
            if self.print:
                print("no valid mode")
            return None

        return im, self.get_label(im, target_label)

    def attack_on_label(self, im_url, save_url, target_label):
        im = url_to_torch(im_url)
        labels = save_and_query(torch_to_array(im[-1]), save_url).reshape(1, LABEL_AMOUNT)
        stop = False
        steps = 0

        mse = (labels[-1][target_label] - self.get_label([im[-1]], target_label)) ** 2
        while stop is False:
            steps += 1
            if self.print:
                print()
                print("step: " + str(steps))
                print("MSE White vs Black Box before retraining:")
                print(mse)
            if self.retrain_mode == "last":
                self.retrain(np.array([im[-1]]), np.array([labels[-1]]), target_label)
            elif self.retrain_mode == "full":
                self.retrain(np.array(im), np.array(labels), target_label)
            elif self.retrain_mode == "none":
                stop = True
            else:
                if self.print:
                    print("Not a valid retrain method. Use \"full\" or \"last\" "
                          "for retraining. Running once without retraining.")
                break

            if self.fgsm_restart == "original":
                advers = self._create_advers([im[0]], target_label, [im[0]])
            elif self.fgsm_restart == "last":
                advers = self._create_advers([im[-1]], target_label, [im[0]])
            else:
                if self.print:
                    print("start should be \"original\" or \"last\" ")
                break
            im = np.concatenate((im, advers))

            new_label = save_and_query(torch_to_array(im[-1]), save_url).reshape(1, LABEL_AMOUNT)
            labels = np.concatenate((labels, new_label))
            target = labels[-1][target_label]
            if self.print:
                print("probability: Blackbox")
                print(target)
                print()
            mse = (target - self.get_label([im[-1]], target_label))**2

            if target > self.target_threshold:
                stop = True
                if self.print:
                    print("found adversarial example")
            elif steps >= self.restart_max_amount or mse < self.restart_accuracy_bound:
                stop = True
                if self.print:
                    print("convergence to target confidence failed. relax bounds,"
                          " increase loop length or start with another image!")
                if self.always_save is False:
                    try:
                        remove(save_url)
                    except FileNotFoundError:
                        pass
        # noinspection PyUnboundLocalVariable
        return target

    def simple_attack(self, im_url, save_url):
        target_label = np.argmax(np.array(query_to_labels(im_url)))
        if self.print:
            print("attacking label " + str(target_label))
        return self.attack_on_label(im_url, save_url, target_label)

    def simple_batch_attack(self, im_folder, save_folder, title=""):
        imlist = glob(im_folder + "/*")
        if len(imlist) == 0:
            raise Exception("im_folder is empty!")
        try:
            imlist.remove(im_folder + "\\desktop.ini")
        except ValueError:
            pass
        for i in range(len(imlist)):
            if imlist[i].split(".")[-1] != "png":
                continue
            else:
                save_name = save_folder + "/" + imlist[i].replace("\\", "/").split("/")[-1]
                save_name = save_name.split(".")[0]
                save_name = save_name + title + ".png"
                self.simple_attack(imlist[i], save_name)
                sleep(self.restart_max_amount + 2)
        return None

    def batch_attack_on_label(self, im_folder,save_folder, target_label, title=""):
        imlist = glob(im_folder + "/*")
        if len(imlist) == 0:
            raise Exception("im_folder is empty!")
        try:
            imlist.remove(im_folder + "\\desktop.ini")
        except ValueError:
            pass
        for i in range(len(imlist)):
            if imlist[i].split(".")[-1] != "png":
                continue
            else:
                save_name = save_folder + "/" + imlist[i].replace("\\", "/").split("/")[-1]
                save_name = save_name.split(".")[0]
                save_name = save_name + title + ".png"
                self.attack_on_label(imlist[i], save_name,
                                     target_label)
                sleep(self.restart_max_amount + 2)
        return None

    def reload_model(self, model):
        if model is None:
            self.model = create_whitebox(self.device)
        else:
            self.model = model

    @staticmethod
    def preview_im(im_url):
        print("Labels with highest confidence:")
        print(query_names(im_url))
        print("try attacking one of those labels!")
        return query_names(im_url)
