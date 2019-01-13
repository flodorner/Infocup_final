Die Software basiert auf Python-skripten und besteht aus einer GUI zur einfachen Erstellung 
von Bildern zur T�uschung des gegebenen neuronalen Netzes sowie einigen Funktionen zur automatischen 
Erstellung von t�uschenden Stickern und zur automatischen Erstellung t�uschender Bilder aus 
einem gegebenen Bild. 

F�r die Software werden neben bereits in Python vorinstallierten Bibliotheken die Bibliotheken
numpy, requests, torch, pillow und Tk ben�tigt.
Am einfachsten ist die Installation auf der Referenzplattform durchzuf�hren, indem in folgender Reihenfolge
die Installationsanweisungen auf den jeweiligen Projektseiten befolgt werden:
1. Installieren von Python 3.7.1, numpy, requests, tk und pillow in der Anaconda Distribution f�r Linux: https://www.anaconda.com/download/#download und folgen der Installationsanweisungen
(Herunterladen und ausf�hren des Shell-Skriptes und setzen des Pfades (export PATH=~/anaconda3/bin:$PATH).
2. Installieren von Pytorch 1.0 via conda durch befolgen der Anweisungen auf https://pytorch.org/ (oder ohne Cuda in der Kommandozeile: conda install pytorch-cpu torchvision-cpu -c pytorch) 
Sollte Cuda auf dem System vorhanden sein empfiehlt sich eine Installation mit der entsprechenden Cuda-Version.
3. Die Gui kann nun �ber die Kommandozeile via python gui.py aufgerufen werden (im entsprechenden Ordner)
4. Die Generation von Stickern und die strukturierte Generation von t�uschendem Rauschen passieren �ber Instanzmethoden der Klassen StickerGenerator in sticker.py bzw. FGSM in fgsm.py.
Um eine der Methoden auszuf�hren wird im entsprechenden Ordner eine python Konsole ge�ffnet, die jeweilige Klasse importiert (from sticker import StickerGenerator / from fgsm import FGSM) und eine Instanz erstellt
(Instanz = StickerGenerator() bzw FGSM()). Durch Instanz.methode() (z.B. Instanz.make_sticker())lassen sich nun die entsprechenden Methoden benutzen. 
S�mtliche Bildmethoden wurden Prim�r mit 64x64 "RGB"-png dateien getestet, "RGBA" sollte aber eigentlich keine Probleme darstellen. Das Verwenden andere Formate kann potentiell zu Fehlern f�hren. 

Beschreibung der Komponenten:
gui:
	Das Aufrufen von gui.py in Python �ffnet die Benutzeroberfl�che. 
	Links in der Mitte wird das Ausgangsbild und rechts in der Mitte das neue Bild angezeigt. 

	�ber den Load image Button wird ein Ladedialog ge�ffnet der es erlaubt ein Bild einzuladen. Dieses erscheint
	zun�chst in beiden Fenstern. Weiterhin wird nach laden des Bildes unter dem Bild die Klasse mit der h�chsten 
	Konfidenz der Blackbox f�r das Bild, sowie die Konfidenz angezeigt.

	Auf der rechten Seite daneben steht die Konfidenz, die unsere Whietebox dem rechten Bild zuordnet.
	Durch den Retrain Whitebox Knopf etwas weiter oben l�sst sich die Whitebox anpassen, 
	um die Diskrepanz in den Konfidenzen zu reduzieren. 

	Durch den Add Noise Knopf kann Rauschen zu dem Bild auf der rechten Seite hinzugef�gt werden,
	das die Konfidenz der Black und der Whitebox in die gew�hlte Klasse  im Normalfall erh�hen sollte.
	Dar�ber k�nnen die Klasse (Label), die St�rke des zugef�gten Rauschens (Noise Magnitude), 
	sowie der maximale Abstand zwischen dem originalen und ver�nderten Bild pro Pixel und Kanal (Noise Bound) 
	festgelegt werden.

	Das resultierende Bild kann �ber den Save image Dialog links oben gespeichert werden.
	Durch den Reset image Knopf wir das ver�nderte Bild wieder auf das original zur�ckgesetzt.

	Durch den Add sticker und Add transparent Sticker Knopf k�nnen mit der sticker.py Klasse generierte Sticker 
	auf das Bild geklebt werden, entweder intransparent, oder aber Kanalweise ("transparent"). 
	Dabei ist zu beachten, dass die "transparenten" Sticker auf wei�en Hintergrund nicht sichtbar sind 
	(Das liegt daran, dass der Angriff f�r maximale Effektivit�t nur niedrige Farbkanalwerte durch Hohe ersetzt
	und nicht umgekehrt).
	Will man den Effekt eines transparenten Stickers auf wei�em Hintergrund dennoch erreichen,
	l�sst sich dies durch Anwenden des nicht transparenten Sticker, einstellen von noise magnitude = 0 sowie einem
	hohen noise bound und dr�cken des add noise Knopes erreichen. Die durch den Sticker erh�hten Farbkan�le werden
	dann reduziert, um nah genug am Original zu sein, was dem gew�nschten Effekt gleichkommt.


sticker:
	Die Klasse StickerGenerator kann mit den Werten pixelsize und fringe initialisiert werden (standardeinstellungen: 3, 17), die die Gr��e der Pixelbl�cke, und deren Abstand zum Rand auf beiden Seiten bestimmen. 
	Dabei sollte die Bildgr��e Minus 2 Mal fringe durch Pixelsize teilbar sein.
		Mit der (Instanz)Methode sticker_batch k�nnen nun Sticker generiert werden, die in dem Subverzeichnis f�r die entsprechende Klasse gespeichert werden. Mit dem optionalen Argument 	
		title (ohne Angabe "") kann ein Titel f�r den Stickerbatch bestimmt werden. F�r die einzlnen Sticker werden der Pixel_threshold und die Konfidenz an den Namen angh�ngt. 
		Pixel_threshold (ohne Angabe 0.01)  gibt vor, wie viel Konfidenzgewinn ein Pixel bringen muss um in den Sticker aufgenommen zu werden.
		Es werden diejenigen Sticker gespeichert, die auf schwarzem Hintergrund eine Konfidenz von mehr als save_threshold (ohne Angabe 0.9) f�r die zugeh�rige Klasse 

		Mit der Methode sticker_attack kann nun ein Sticker auf ein Bild geklebt und dieses gespeichert werden.
		Als input erh�lt die Methode die Url des Bildes und eine Ziel-Url zum Abspeichern als strings (in Anf�hrungszeichen). Optional k�nnen eine bestimmte Zielklasse ("label", als Zahl codiert, siehe unten), 
		oder ein bestimmter Sticker als Url ("sticker_url") angegeben werden. Durch setzen des optionalen Argumentes "mode" auf "transparent" kann ein Sticker zudem transparent anstatt
		f�llend aufgeklebt werden.
fgsm:
	Die Klasse FGSM kann mit einem Pytorch model als Whitebox, sowie einem Parameter cuda, der bestimmt ob cuda, falls vorhanden zur Beschleunigung genutzt werden soll, initialisiert werden.
	Ohne Angabe eines models wird die von uns trainierte Whitebox verwendet. 
		Die Methode attack_on_label erh�lt die Url f�r ein Basisbild, eine Url zum speichern, sowie ein label, zu dessen Erkennung das angegriffene Netz gebracht werden soll. 
		Danach wird eine Iterierte fast gradient sign Attacke inklusive Anpassung der Whitebox ausgehend von dem Basisbild durchgef�hrt und das Ergebnis gespeichert. Die genauen Parameter der Attacke
		lassen sich durch �ndern von FGSM_SPECS in config.py spezifizieren (danach ist ein erneutes importieren der Klasse in python n�tig!). Mehr dazu weiter unten.
		
		Die Methode preview_im zeigt bei Eingabe einer g�ltigen URL f�r ein Bild die f�nf Klassen mir der h�chsten Konfidenz, f�r die ein Angriff also wahrscheinlich am erfolgreichsten ist, sowie die 
		entsprechenden numerischen Labels und die Konfidenzen an.
	
		Die Methode simple_attack �bernimmt die Auswahl des Labels selbst und speichert, gegeben ein Basisbild (als Url) und eine Ziel-Url, das t�uschende Bild in der Ziel-Url.
		
		Unter Umst�nden kann die Qualit�t der Whitebox durch zu viel Training auf nicht repr�sentativen Regionen w�hrend des Generationsprozesses abnehmen. Um keine neue Instanz erstellen zu m�ssen,
		wenn das Modell auf den von uns trainierten Stand zur�ckgesetzt werden soll, gibt es eine reload_model Methode, die das Modell neu l�dt. Dieser kann auch ein neues Modell als argument gegeben werden, um
		das verwendete Modell schnell zu wechseln.  

	Die Parameter f�r die FGSM-Attacke befinden sich wie bereits erw�hnt im python-dictionary FGSM_SPECS, welches in config.py gespeichert ist. Im folgenden die Default-Einstellungen und eine Erkl�rung der Parameter:
	"mode": "l_inf" Gibt die Form der Projektion nach jeder Iteration an. "l_inf": Maximumsnorm "l_2": Euklidnorm "simple": keine Projektion. 
   	"bound": 10  Gibt den Maximalwert der gew�hlten Norm an, auf den der Abstand dann zur�ck projiziert wird. F�r "l_2" empfiehlt sich ein wert um 1000. 
   	"magnitude": 1 Gibt das alpha/die Schrittgr��e f�r die FGSM-Iteration an.
   	"max_fgsm_iterations": 25 Gibt die Maximale Schrittanzahl in der FGSM-Iteration an. 
    	"target_threshold": 0.99 Schwellenwert f�r die Konfidenz. Wenn dieser w�hrend der FGSM-Iteration �berschritten wird, unterbricht diese und pr�ft den Abstand zwischen Black- und Whitebox und trainiert dann entweder 
				die Whitebox nach, oder beendet bei Erfolg die Schleife.

   	"fgsm_restart": "last" Bestimmt, ob nach dem adjustieren der Whitebox von vorne ("original") oder mit dem Endergebnis der letzten Iteration ("last") begonnen werden soll.
    	"restart_max_amount": 10 Bestimmt die Maximale Anzahl an Versuchen, die Whitebox zu adjustieren.
    	"restart_accuracy_bound": 0.0001 Betr�gt der Quadrierte Abstand zwischen Vorhersage von White- und Blackbox am ende einer FGSM-Iterarion weniger als diese Schwelle, wird die Iteration abgebrochen.

   	"retrain_mode": "last" Bestimmt, ob nur f�r die letzte FGSM-Iteration adjustiert werden soll ("last"), oder ob auch die Ergebnisse der vorigen Iterationen einbezogen werden sollen ("full").
    	"retrain_lr": 0.00001 Bestimmt die Lerngeschwindigkeit beim Adjustieren. Niedrige werte verlangsamen die Anpassung, hohe f�hren zu lokaler �beranpassung oder sogar zu schlechterer Konvergenz.
    	"retrain_max_gradient_steps": 10 Bestimmt die Anzahl der Optimierungsschritte beim Adjustieren. Effekte �hlich wie bei der Lerngeschwindigkeit, hohe Werte verschlechtern die Konvergenz hier nicht, ben�tigen aber Zeit.
    	"retrain_threshold": 0.0001 Bestimmt den Schwellenwert f�r den Quadratischen Abstand zwischen der Vorhersage von Black- und Whitebox, an dem das Adjustieren abgebrochen wird.
    	"always_save": True Bestimmt, ob Endergebnisse, deren Konfidenz unter target_threshold liegt trotzdem gespeichert werden sollen.
    	"print": True Bestimmt, ob Informationen, wie die Konfidenzen bei Zwischenschritten, ausgegeben werden sollen.

		
Weitere Config-Parameter:
	URL = 'https://phinau.de/trasi' , KEY = 'ut6ohb7ZahV9tahjeikoo1eeFaev1aef' (Abfrage der Blackbox)
	STICKER_DIRECTORY = "Quickstick" Superverzeichnis f�r das Abspeichern der Sticker.
	LABEL_AMOUNT = 43 Anzahl der Klassen
	IMAGE_SIZE = 64 Gr��e der Bilder in Pixeln


Liste der Klassen und der zugordneten numerischen Label: (CLASSNAMEDICT)
		 'Zul�ssige H�chstgeschwindigkeit (20)': 0,
  		 'Zul�ssige H�chstgeschwindigkeit (30)': 1,
                 'Zul�ssige H�chstgeschwindigkeit (50)': 2,
                 'Zul�ssige H�chstgeschwindigkeit (60)': 3,
                 'Zul�ssige H�chstgeschwindigkeit (70)': 4,
                 'Zul�ssige H�chstgeschwindigkeit (80)': 5,
                 'Ende der Geschwindigkeitsbegrenzung (80)': 6,
                 'Zul�ssige H�chstgeschwindigkeit (100)': 7,
                 'Zul�ssige H�chstgeschwindigkeit (120)': 8,
                 '�berholverbot f�r Kraftfahrzeuge aller Art': 9,
                 '�berholverbot f�r Kraftfahrzeuge mit einer zul�ssigen Gesamtmasse �ber 3,5t': 10,
                 'Einmalige Vorfahrt': 11,
                 'Vorfahrt': 12,
                 'Vorfahrt gew�hren': 13,
                 'Stoppschild': 14,
                 'Verbot f�r Fahrzeuge aller Art': 15,
                 'Verbot f�r Kraftfahrzeuge mit einer zul�ssigen Gesamtmasse von 3,5t': 16,
                 'Verbot der Einfahrt': 17,
                 'Gefahrenstelle': 18,
                 'Kurve (links)': 19,
                 'Kurve (rechts)': 20,
                 'Doppelkurve (zun�chst links)': 21,
                 'Unebene Fahrbahn': 22,
                 'Schleudergefahr bei N�sse oder Schmutz': 23,
                 'Fahrbahnverengung (rechts)': 24,
                 'Baustelle': 25,
                 'Lichtzeichenanlage': 26,
                 'Fu�g�nger': 27,
                 'Kinder': 28,
                 'Fahrradfahrer': 29,
                 'Schnee- oder Eisgl�tte': 30,
                 'Wildwechsel': 31,
                 'Ende aller Streckenverbote': 32,
                 'Ausschlie�lich rechts': 33,
                 'Ausschlie�lich links': 34,
                 'Ausschlie�lich geradeaus': 35,
                 'Ausschlie�lich geradeaus oder rechts': 36,
                 'Ausschlie�lich geradeaus oder links': 37,
                 'Rechts vorbei': 38,
                 'Links vorbei': 39,
                 'Kreisverkehr': 40,
                 'Ende des �berholverbotes f�r Kraftfahrzeuge aller Art': 41,
                 'Ende des �berholverbotes f�r Kraftfahrzeuge mit einer zul�ssigen Gesamtmasse �ber 3,5t': 42