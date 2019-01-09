import torch
from torch import optim, nn

from config import fgsm_specs as specs
from utilities import torch_to_saveable, url_to_torch, save_and_query, query_to_labels, query_with_labelnums
from distill import create_distilled
import numpy as np


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


class FGSM:
    def __init__(self, model=None, cuda=True):

        if torch.cuda.is_available() and cuda is True:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        if model is None:
            self.model = create_distilled(self.device)
        else:
            self.model = model

        self.mode = specs["mode"]
        self.bound = specs["bound"]
        self.magnitude = specs["magnitude"]
        self.max_fgsm_iterations = specs["max_fgsm_iterations"]
        self.target_threshold = specs["target_threshold"]

        self.fgsm_restart = specs["fgsm_restart"]
        self.restart_max_amount = specs["restart_max_amount"]
        self.restart_accuracy_bound = specs["restart_accuracy_bound"]

        self.retrain_mode = specs["retrain_mode"]
        self.retrain_lr = specs["retrain_lr"]
        self.retrain_max_gradient_steps = specs["retrain_max_gradient_steps"]
        self.retrain_threshold = specs["retrain_threshold"]

    def reload_model(self, model):
        if model is None:
            self.model = create_distilled(self.device)
        else:
            self.model = model

    def get_label(self, im, target_label):
        self.model.eval()
        output = self.model(torch.tensor(im).to(self.device))
        label = torch.take(output, torch.tensor(target_label).to(self.device)).item()
        return label

    def get_gradient(self, im, target_label):
        self.model.eval()
        im_tensor = torch.tensor(im).to(self.device)
        im_tensor.requires_grad = True
        output = self.model(im_tensor)
        loss = torch.take(output, torch.tensor(target_label).to(self.device))
        loss.backward()
        im_grad = im_tensor.grad

        return im_grad

    def fastgrad_step(self, im, target_label, base):
        im_grad = self.get_gradient(im, target_label)
        im = im + self.magnitude * np.sign(im_grad.cpu().detach().numpy())
        if self.mode == "simple":
            pass
        elif self.mode == "l_2":
            im = project_l_2(im, base, self.bound)
        elif self.mode == "l_inf":
            im = project_l_inf(im, base, self.bound)
        else:
            print("no valid mode")
            return None

        return im, self.get_label(im, target_label)

    def train_on_label(self, images, labels, target_label):
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

    def adapt(self, im, label, target_label):
        i = 0
        error = 1
        while error > self.retrain_threshold and i < self.retrain_max_gradient_steps:
            i = i + 1
            error = self.train_on_label(im, label, target_label)
        print("MSE White vs Black Box after retraining:")
        print(error)
        return i, error

    def create_advers(self, im, target_label, base):
        prob = 0
        steps = 0
        while prob < self.target_threshold and steps < self.max_fgsm_iterations:
            steps += 1
            im, prob = self.fastgrad_step(im, target_label, base)
        print("probability: Whitebox")
        print(prob)
        return im

    def attack_on_label(self, im_url, save_url, target_label,):
        print()
        im = url_to_torch(im_url)
        labels = save_and_query(torch_to_saveable(im[-1]), save_url).reshape(1, 43)
        stop = False
        steps = 0

        mse = (labels[-1][target_label] - self.get_label([im[-1]], target_label)) ** 2
        while stop is False:
            steps += 1
            print("step: " + str(steps))
            print("MSE White vs Black Box before retraining:")
            print(mse)
            if self.retrain_mode == "last":
                self.adapt(np.array([im[-1]]), np.array([labels[-1]]), target_label)
            elif self.retrain_mode == "full":
                self.adapt(self.model, self.device, np.array(im), np.array(labels), target_label)
            elif self.retrain_mode == "none":
                stop = True
            else:
                print("Not a valid retrain method. Use \"full\" or \"last\" "
                      "for retraining. Running once without retraining.")
                break

            if self.fgsm_restart == "original":
                advers = self.create_advers([im[0]], target_label, [im[0]]).reshape((1, 3, 64, 64))
            elif self.fgsm_restart == "last":
                advers = self.create_advers([im[-1]], target_label, [im[0]]).reshape((1, 3, 64, 64))
            else:
                print("start should be \"original\" or \"last\" ")
            im = np.concatenate((im, advers))

            new_label = save_and_query(torch_to_saveable(im[-1]), save_url).reshape(1, 43)
            labels = np.concatenate((labels, new_label))
            target = labels[-1][target_label]
            print("probability: Blackbox")
            print(target)
            mse = (target - self.get_label([im[-1]], target_label))**2
            print()
            if target > self.target_threshold:
                stop = True
                print("found adversarial example")
            if steps >= self.restart_max_amount  or mse < self.restart_accuracy_bound:
                stop = True
                print("convergence failed. relax bounds, increase loop length or start with another image!")
        return None

    def preview_im(self, im_url):
        print("Labels with highest confidence:")
        query_with_labelnums(im_url)
        print("try attacking one of those labels!")

    def simple_attack(self, im_url, save_url):
        target_label = np.argmax(np.array(query_to_labels(im_url)))
        print("attacking label " + str(target_label))
        self.attack_on_label(im_url, save_url, target_label)
        return None