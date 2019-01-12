Die Software basiert auf Python-skripten und besteht aus einer GUI zur einfachen Erstellung 
von Bildern zur Täuschung des gegebenen neuronalen Netzes sowie Skripten zur automatischen 
Erstellung von täuschenden Stickern und zur automatischen Erstellung täuschender Bilder aus 
einem gegebenen Bild. 

Für die Software werden neben bereits in Python vorinstallierten Bibliotheken die Bibliotheken
numpy, requests, torch, pillow und Tk benötigt.

Am einfachsten ist die Installation auf der Referenzplattform durchzuführen, indem in folgender Reihenfolge
die Installationsanweisungen auf den jeweiligen Projektseiten befolgt werden:
1. Installieren von Anaconda für Linux: https://www.anaconda.com/download/#download und folgen der Installationsanweisungen
2. Installieren von numpy, requests, pillow und Tk via anaconda: 
https://anaconda.org/anaconda/numpy 
https://anaconda.org/anaconda/pillow 
https://anaconda.org/anaconda/requests
https://anaconda.org/anaconda/tk
3. Installieren von Pytorch via conda durch befolgen der Anweisungen auf https://pytorch.org/
Sollte Cuda auf dem System vorhanden sein empfiehlt sich eine Installation mit der entsprechenden Cuda-Version.

Beschreibung der Komponenten:
Das Aufrufen von gui.py in Python öffnet die Benutzeroberfläche. 
Links in der Mitte wird das Ausgangsbild und rechts in der Mitte das neue Bild angezeigt. 

Über den Load image Button wird ein Ladedialog geöffnet der es erlaubt ein Bild einzuladen. Dieses erscheint
zunächst in beiden Fenstern. Weiterhin wird nach laden des Bildes unter dem Bild die Klasse mit der höchsten 
Konfidenz der Blackbox für das Bild, sowie die Konfidenz angezeigt.

Auf der rechten Seite daneben steht die Konfidenz, die unsere Whietebox dem rechten Bild zuordnet.
Durch den Retrain Whitebox Knopf etwas weiter oben lässt sich die Whitebox anpassen, 
um die Diskrepanz in den Konfidenzen zu reduzieren. 

Durch den Add Noise Knopf kann Rauschen zu dem Bild auf der rechten Seite hintugefügt werden,
das die Konfidenz der Black und der Whitebox in die gewählte Klasse  im Normalfall erhöhen sollte.
Darüber können die Klasse (Label), die Stärke des zugefügten Rauschens (Noise Magnitude), 
sowie der maximale Abstand zwischen dem originalen und veränderten Bild pro Pixel und Kanal (Noise Bound) 
festgelegt werden.

Das resultierende Bild kann über den Save image Dialog links oben gespeichert werden.
Durch den Reset image Knopf wir das veränderte Bild wieder auf das original zurückgesetzt.

Durch den Add sticker und Add transparent Sticker Knopf können mit der sticker.py Klasse generierte Sticker 
auf das Bild geklebt werden, entweder intransparent, oder aber Kanalalweise ("transparent"). 
Dabei ist zu beachten, dass die "transparenten" Sticker auf weißen Hintergrund nicht sichtbar sind 
(Das liegt daran, dass der Angriff für maximale Effektivität nur niedrige Farbkanalwerte durch Hohe ersetzt
und nicht umgekehrt).
Will man den Effekt eines transparenten Stickers auf weißem Hintergrund dennoch erreichen,
lässt sich dies durch Anwenden des nichttransparenten Sticker, einstellen von noise magnitude = 0 sowie einem
Hohen noise bound und drücken des add noise Knopes erreichen. Die durch den Sticker erhöhten Farbkanäle werden
dann reduziert, um nah genug am Original zu sein, was dem gewünschten Effekt gleichkommt.






