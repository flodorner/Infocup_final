from utilities import torch_to_saveable,save_and_query
from distill import create_distilled()

#Update self dict from dict in config
#Methode fürr plausibel zu updatende
#load model instead of having it as argument

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
    def __init__(self,device,model=None):
        if model is None:
            self.model=model
        else:
            model=create_distilled()
        self.device=device

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

    def fastgrad_step(self, im, target_label, base, mode, bound, magnitude=1):
        im_grad = get_gradient(self.model, self.device, im, target_label)

        im = im + magnitude * np.sign(im_grad.cpu().detach().numpy())

        if mode == "simple":
            pass
        elif mode == "l_2":
            im = self.project_l_2(im, base, bound)
        elif mode == "l_inf":
            im = self.project_l_inf(im, base, bound)
        else:
            print("no valid mode")
            return None

        return im, self.get_label(im, target_label)



    def train_on_label(self, images, labels, target_label):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
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
        return loss.item()


    def adapt(self, image, label, target_label, threshold=0.0001, max_train_steps=50):
        i = 0
        error = 1
        while error > threshold and i < max_train_steps == 100:
            i = i + 1
            error = self.train_on_label(image, label, target_label)
        print("MSE White vs Black Box")
        print(error)
        return i, error


    def create_advers(self, im, target_label, base, mode, bound, maxsteps=25, target_threshold=0.9):
        prob = 0
        steps = 0
        while prob < target_threshold and steps < maxsteps:
            steps += 1
            im, prob = self.fastgrad_step(im, target_label, base, mode, bound)
        print("probability: Whitebox")
        print(prob)
        return im

    def simple_attack(self, im, labels, target_label, save_url, mode="simple", bound=None, start="original",
                      target_threshold=0.99, retrain="last", retrain_threshold=0.0001, max_inner_steps=25,
                      max_outer_steps=5, max_train_steps=50):
        if bound is None:
            if mode == "l_inf":
                bound = 10
            if mode == "l_2":
                bound = 1000
        stop = False
        steps = 0
        while stop is False:
            steps += 1
            if retrain == "last":
                self.adapt(self.model, self.device, np.array([im[-1]]), np.array([labels[-1]]), target_label, retrain_threshold,
                      max_train_steps)
            elif retrain == "full":
                self.adapt(self.model, self.device, np.array(im), np.array(labels), target_label, retrain_threshold, max_train_steps)
            elif retrain == "none":
                stop = True
            else:
                print("Not a valid retrain method. Running once without retrain")
                stop = True
            if start == "original":
                advers = self.create_advers(self.model, self.device, [im[0]], target_label, [im[0]], mode, bound, maxsteps=max_inner_steps,
                                       target_threshold=target_threshold).reshape((1, 3, 64, 64))
            elif start == "last":
                advers = self.create_advers(self.model, self.device, [im[-1]], target_label, [im[0]], mode, bound,
                                       maxsteps=max_inner_steps, target_threshold=target_threshold).reshape((1, 3, 64, 64))
            else:
                print("start should be \"original\" or \"pass\" ")
                break
            im = np.concatenate((im, advers))

            new_label = save_and_query(torch_to_saveable(im[-1]), save_url).reshape(1, 43)
            labels = np.concatenate((labels, new_label))
            target = labels[-1][target_label]
            print("probability: Blackbox")
            print(target)
            if target > target_threshold:
                stop = True
                print("found adversarial example")
            if steps > max_outer_steps:
                stop = True
                print("convergence failed. relax bounds, increase loop length or start with another image!")
        return None














