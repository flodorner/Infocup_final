from config import *
from utilities import url_to_array, url_to_torch, torch_to_array, query_to_labels, ServerError
from sticker import stick, stick_trans
from fgsm import FGSM
from GAN import create_image

from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from os import path
import numpy as np
import textwrap


class AdversarialStudio:

    def __init__(self):

        self.Root = Tk()
        self.Root.title('Adversarial Studio')

        self.Root.resizable(False, False)

        self.Generator = FGSM()
        self.Generator.print = False

        self.Original_description = Label(self.Root, text="Original Image:")
        self.Original_text = Label(self.Root, text="")
        self.orig_img_PIL = Image.open("GUI/grey.png").resize((IMAGE_SIZE, IMAGE_SIZE))
        orig_img = ImageTk.PhotoImage(self.orig_img_PIL.resize((256, 256)))
        self.Original_image = Label(self.Root, image=orig_img)

        self.Edited_description = Label(self.Root, text="Edited Image:")
        self.edit_img_PIL = self.orig_img_PIL.copy()
        self.Edited_text = Label(self.Root, text="")
        edit_img = ImageTk.PhotoImage(self.edit_img_PIL.resize((256, 256)))
        self.Edited_image = Label(self.Root, image=edit_img)

        self.Whitebox_description = Label(self.Root, text="")
        self.Whitebox_text = Label(self.Root, text="")

        self.Top = Frame(self.Root)
        self.Load_Button = Button(self.Top, text="Load image", command=self._open_button)
        self.Save_Button = Button(self.Top, text="Save image", command=self._save_button)
        self.Reset_Button = Button(self.Top, text="Reset current image", command=self._reset_button)
        self.Sticker_Button_full = Button(self.Top, text="Add sticker", command=lambda: self._sticker_button("full"))
        self.Sticker_Button_trans = Button(self.Top, text="Add transparent sticker",
                                           command=lambda: self._sticker_button("trans"))
        self.Generate_button = Button(self.Top, text="Generate Face", command=self._generate_button)

        self.Interface = Frame(self.Root)
        self.Buttons = Frame(self.Interface)
        self.Noise_button = Button(self.Buttons, text="Add Noise", command=self._advers_attack)
        self.Retrain_button = Button(self.Buttons, text="Retrain Whitebox", command=self._retrain_whitebox)

        self.Label_text = Label(self.Interface, text="Target Label:")
        self.Label_param = Entry(self.Interface,
                                 validate="focusout", validatecommand=self._islabel,
                                 invalidcommand=self._resetlabel)
        self.Label_param.insert(0, "4")

        self.Bound_text = Label(self.Interface, text="Noise Bound:")
        self.Bound_param = Entry(self.Interface, validate="focusout", validatecommand=self._isbound,
                                 invalidcommand=self._resetbound)
        self.Bound_param.insert(0, FGSM_SPECS["bound"])

        self.Magnitude_text = Label(self.Interface, text="Noise Magnitude:")
        self.Magnitude_param = Entry(self.Interface, validate="focusout", validatecommand=self._ismagnitude,
                                     invalidcommand=self._resetmagnitude)
        self.Magnitude_param.insert(0, FGSM_SPECS["magnitude"])
        
        self.Top.grid(row=0, columnspan=6, sticky=W)
        self.Load_Button.grid(row=0, column=0, sticky=W)
        self.Save_Button.grid(row=0, column=1, sticky=W)
        self.Reset_Button.grid(row=0, column=2, sticky=W)
        self.Sticker_Button_full.grid(row=0, column=3, sticky=W)
        self.Sticker_Button_trans.grid(row=0, column=4, sticky=W)
        self.Generate_button.grid(row=0, column=5, sticky=W)

        self.Original_image.grid(row=1, column=0, rowspan=3, columnspan=3, sticky=W)
        self.Original_description.grid(row=4, column=0, columnspan=3, rowspan=2, sticky=W)
        self.Original_text.grid(row=6, column=0, columnspan=3, rowspan=2, sticky=W)

        self.Edited_image.grid(row=1, column=3, rowspan=3, columnspan=3, sticky=W)
        self.Edited_description.grid(row=4, column=3, columnspan=3, rowspan=2, sticky=W)
        self.Edited_text.grid(row=6, column=3, columnspan=3, rowspan=2, sticky=W)

        self.Whitebox_description.grid(row=4, column=6, columnspan=3, rowspan=2, sticky=W)
        self.Whitebox_text.grid(row=6, column=6, columnspan=3, rowspan=2, sticky=W)

        self.Interface.grid(row=2, column=7)

        self.Label_text.grid(row=0, column=0, sticky=E)
        self.Label_param.grid(row=0, column=1)
        self.Bound_text.grid(row=1, column=0, sticky=E)
        self.Bound_param.grid(row=1, column=1)
        self.Magnitude_text.grid(row=2, column=0, sticky=E)
        self.Magnitude_param.grid(row=2, column=1)

        self.Buttons.grid(row=4, columnspan=2)
        self.Noise_button.grid(row=0, column=0)
        self.Retrain_button.grid(row=0, column=1)

        self.Root.mainloop()

    def _get_max_label(self):
        self.edit_img_PIL.save("GUI/temp.png")
        try:
            label = np.argmax(query_to_labels("GUI/temp.png"))
        except ServerError:
            label = 4
        return label

    def _get_orig_conf(self):
        self.orig_img_PIL.save("GUI/temp.png")
        try:
            conf = query_to_labels("GUI/temp.png")[int(self.Label_param.get())]
        except ServerError:
            conf = "Error: Try Reloading"
        return conf

    def _get_edit_conf(self):
        self.edit_img_PIL.save("GUI/temp.png")
        try:
            conf = query_to_labels("GUI/temp.png")[int(self.Label_param.get())]
        except ServerError:
            conf = "Error: Try Reloading"
        return conf

    def _islabel(self):
        if not self.Label_param.get().isdigit():
            return False
        elif int(self.Label_param.get()) < LABEL_AMOUNT:
            self._refresh_labels()
            return True
        else:
            return False

    def _isbound(self):
        if not self.Bound_param.get().isdigit():
            return False
        else:
            self.Generator.bound = int(self.Bound_param.get())
            return True

    def _ismagnitude(self):
        if not self.Magnitude_param.get().isdigit():
            return False
        else:
            self.Generator.magnitude = int(self.Magnitude_param.get())
            return True

    def _resetlabel(self, string="4"):
        self.Label_param.delete(0, 'end')
        self.Label_param.insert(0, string)

    def _resetbound(self):
        self.Bound_param.delete(0, 'end')
        self.Bound_param.insert(0, FGSM_SPECS["bound"])
        self.Generator.bound = int(self.Bound_param.get())

    def _resetmagnitude(self):
        self.Magnitude_param.delete(0, 'end')
        self.Magnitude_param.insert(0, FGSM_SPECS["magnitude"])
        self.Generator.magnitude = int(self.Magnitude_param.get())

    def _add_whitebox_info(self):
        self.Whitebox_description.configure(text="Whitebox prediction: ")
        self.edit_img_PIL.save("GUI/temp.png")
        im = url_to_torch("GUI/temp.png")

        text = str("%.2f" % self.Generator.get_label(im, int(self.Label_param.get())))
        self.Whitebox_text.configure(text=text)

    def _refresh_labels(self):
        conf = self._get_orig_conf()
        text = REVERSE_CLASSNAMEDICT[int(self.Label_param.get())] + ": " + str("%.2f" % conf)
        text = textwrap.fill(text, 45)
        self.Original_text.configure(text=text)
        conf = self._get_edit_conf()
        text = REVERSE_CLASSNAMEDICT[int(self.Label_param.get())] + ": " + str("%.2f" % conf)
        text = textwrap.fill(text, 45)
        self.Edited_text.configure(text=text)

    def _open_button(self):
        filename = filedialog.askopenfilename()
        open_path = filename

        self.orig_img_PIL = Image.open(open_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        orig_img = ImageTk.PhotoImage(self.orig_img_PIL.resize((256, 256)))
        self.Original_image.configure(image=orig_img)
        self.Original_image.image = orig_img

        self.edit_img_PIL = Image.open(open_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        edit_img = ImageTk.PhotoImage(self.edit_img_PIL.resize((256, 256)))
        self.Edited_image.configure(image=edit_img)
        self.Edited_image.image = edit_img

        self._resetlabel(str(self._get_max_label()))
        self._refresh_labels()
        self._add_whitebox_info()

    def _reset_button(self):
        self.edit_img_PIL = self.orig_img_PIL.copy()
        edit_img = ImageTk.PhotoImage(self.edit_img_PIL.resize((256, 256)))
        self.Edited_image.configure(image=edit_img)
        self.Edited_image.image = edit_img

        self._refresh_labels()
        self._add_whitebox_info()

    def _save_button(self):
        filename = filedialog.asksaveasfilename(initialdir="/", filetypes=[("png files", "*.png")])
        self.edit_img_PIL.save(filename+".png")

    def _sticker_button(self, mode="full"):
        filename = filedialog.askopenfilename(initialdir=path.abspath(STICKER_DIRECTORY)+"/"+self.Label_param.get())
        start = np.asarray(self.edit_img_PIL.convert("RGB"))
        sticker = url_to_array(filename)
        if mode == "full":
            self.edit_img_PIL = Image.fromarray(stick(start, sticker))
        elif mode == "trans":
            self.edit_img_PIL = Image.fromarray(stick_trans(start, sticker))
        edit_img = ImageTk.PhotoImage(self.edit_img_PIL.resize((256, 256)))
        self.Edited_image.configure(image=edit_img)
        self.Edited_image.image = edit_img

        self._resetlabel(str(self._get_max_label()))
        self._refresh_labels()
        self._add_whitebox_info()

    def _generate_button(self):

        self.orig_img_PIL = Image.fromarray(torch_to_array(create_image())).resize((IMAGE_SIZE, IMAGE_SIZE))
        orig_img = ImageTk.PhotoImage(self.orig_img_PIL.resize((256, 256)))
        self.Original_image.configure(image=orig_img)
        self.Original_image.image = orig_img

        self.edit_img_PIL = self.orig_img_PIL.copy()
        edit_img = ImageTk.PhotoImage(self.edit_img_PIL.resize((256, 256)))
        self.Edited_image.configure(image=edit_img)
        self.Edited_image.image = edit_img

        self._resetlabel(str(self._get_max_label()))
        self._refresh_labels()
        self._add_whitebox_info()

    def _retrain_whitebox(self):
        self.edit_img_PIL.save("GUI/temp.png")
        label = query_to_labels("GUI/temp.png")
        im = url_to_torch("GUI/temp.png")
        self.Generator.retrain(im, np.array([label]), int(self.Label_param.get()))
        self._add_whitebox_info()

    def _advers_attack(self):
        self.edit_img_PIL.save("GUI/temp.png")
        im = url_to_torch("GUI/temp.png")
        self.orig_img_PIL.save("GUI/temp.png")
        base = url_to_torch("GUI/temp.png")

        advers_array, new_label = self.Generator.fastgrad_step(im, int(self.Label_param.get()), base)
        self.edit_img_PIL = Image.fromarray(torch_to_array(advers_array[0]))

        edit_img = ImageTk.PhotoImage(self.edit_img_PIL.resize((256, 256)))
        self.Edited_image.configure(image=edit_img)
        self.Edited_image.image = edit_img

        self.Whitebox_text.configure(text=str("%.2f" % new_label))
        self._refresh_labels()


AdversarialStudio()
