Die Software basiert auf Python-skripten und besteht aus einer GUI zur einfachen Erstellung 
von Bildern zur T�uschung des gegebenen neuronalen Netzes sowie Skripten zur automatischen 
Erstellung von t�uschenden Stickern und zur automatischen Erstellung t�uschender Bilder aus 
einem gegebenen Bild. 

F�r die Software werden neben bereits in Python vorinstallierten Bibliotheken die Bibliotheken
numpy, requests, torch, pillow und Tk ben�tigt.

Am einfachsten ist die Installation auf der Referenzplattform durchzuf�hren, indem in folgender Reihenfolge
die Installationsanweisungen auf den jeweiligen Projektseiten befolgt werden:
1. Installieren von Anaconda f�r Linux: https://www.anaconda.com/download/#download und folgen der Installationsanweisungen
2. Installieren von numpy, requests, pillow und Tk via anaconda: 
https://anaconda.org/anaconda/numpy 
https://anaconda.org/anaconda/pillow 
https://anaconda.org/anaconda/requests
https://anaconda.org/anaconda/tk
3. Installieren von Pytorch via conda durch befolgen der Anweisungen auf https://pytorch.org/
Sollte Cuda auf dem System vorhanden sein empfiehlt sich eine Installation mit der entsprechenden Cuda-Version.

Beschreibung der Komponenten:
Das Aufrufen von gui.py in Python �ffnet die Benutzeroberfl�che. 
Links in der Mitte wird das Ausgangsbild und rechts in der Mitte das neue Bild angezeigt. 

�ber den Load image Button wird ein Ladedialog ge�ffnet der es erlaubt ein Bild einzuladen. Dieses erscheint
zun�chst in beiden Fenstern. Weiterhin wird nach laden des Bildes unter dem Bild die Klasse mit der h�chsten 
Konfidenz der Blackbox f�r das Bild, sowie die Konfidenz angezeigt.

Auf der rechten Seite daneben steht die Konfidenz, die unsere Whietebox dem rechten Bild zuordnet.
Durch den Retrain Whitebox Knopf etwas weiter oben l�sst sich die Whitebox anpassen, 
um die Diskrepanz in den Konfidenzen zu reduzieren. 

Durch den Add Noise Knopf kann Rauschen zu dem Bild auf der rechten Seite hintugef�gt werden,
das die Konfidenz der Black und der Whitebox in die gew�hlte Klasse  im Normalfall erh�hen sollte.
Dar�ber k�nnen die Klasse (Label), die St�rke des zugef�gten Rauschens (Noise Magnitude), 
sowie der maximale Abstand zwischen dem originalen und ver�nderten Bild pro Pixel und Kanal (Noise Bound) 
festgelegt werden.

Das resultierende Bild kann �ber den Save image Dialog links oben gespeichert werden.
Durch den Reset image Knopf wir das ver�nderte Bild wieder auf das original zur�ckgesetzt.

Durch den Add sticker und Add transparent Sticker Knopf k�nnen mit der sticker.py Klasse generierte Sticker 
auf das Bild geklebt werden, entweder intransparent, oder aber Kanalalweise ("transparent"). 
Dabei ist zu beachten, dass die "transparenten" Sticker auf wei�en Hintergrund nicht sichtbar sind 
(Das liegt daran, dass der Angriff f�r maximale Effektivit�t nur niedrige Farbkanalwerte durch Hohe ersetzt
und nicht umgekehrt).
Will man den Effekt eines transparenten Stickers auf wei�em Hintergrund dennoch erreichen,
l�sst sich dies durch Anwenden des nichttransparenten Sticker, einstellen von noise magnitude = 0 sowie einem
Hohen noise bound und dr�cken des add noise Knopes erreichen. Die durch den Sticker erh�hten Farbkan�le werden
dann reduziert, um nah genug am Original zu sein, was dem gew�nschten Effekt gleichkommt.






