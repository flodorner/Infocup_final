### InformatiCup Abgabe Florian Dorner und Julian Stastny

Die Software basiert auf Python und besteht aus einer GUI zur einfachen Erstellung 
von Bildern zur Täuschung des gegebenen neuronalen Netzes sowie einigen Funktionen zur automatischen 
Erstellung von täuschenden Stickern und zur automatischen Erstellung täuschender Bilder aus 
einem gegebenen Bild. Sämtliche Bildmethoden wurden Primär mit 64x64 "RGB"-png dateien getestet, "RGBA" sollte aber eigentlich keine Probleme darstellen. Das Verwenden andere Formate kann potentiell zu Fehlern führen. 

#### Installation (Ubuntu 18.04 LTS AMD64)

Am einfachsten ist die Installation auf der Referenzplattform durchzuführen, indem in folgender Reihenfolge
die Installationsanweisungen auf den jeweiligen Projektseiten befolgt werden:
1. Installation von [Anaconda für Python 3.7](https://www.anaconda.com/download/#linux)  
2. Installation von Pytorch 1.0 mit Cuda in Anaconda
3. Falls nicht vorhanden: Installation von numpy, pandas, requests, tk und pillow in Anaconda

Alternativ kann natürlich auch eine vorhandene Installation von Python 3.7 mit den entsprechenden Bibliotheken verwendet werden. Die Installation sollte auch unter Windows analog funktionieren.

#### Ausführung

```
export PATH="$HOME/anaconda/bin:$PATH"
In dem heruntergeladenen Ordner: python main.py
```

##### Tutorial

Load Image: Erlaubt, ein Bild einzuladen. Für die Nutzung des Generative Adversarial Networks (GAN) sollte ein Bild aus dem Ordner "Faces" geladen werden.

Add Noise: Fügt dem rechten Bild täuschendes FGSM-Rauschen hinzu. Die Stärke und der maximale Abstand zum linken Bild lassen sich durch Eingabe anderer Werte für Magnitude und Bound ändern. Label bestimmt die Zielklasse. 

Retrain White Box: Anpassung der White Box anhand der Vorhersagen der Black Box.

Reset current Image: Setzt das rechte Bild zurück.

Add Generative Noise: Fügt dem rechten Bild durch das GAN generiertes Rauschen hinzu.

Add (transparent) Sticker: klebt den ausgewählten Sticker auf das rechte Bild.

Save Image: Erlaubt, das rechte Bild abzuspeichern.

##### Erweitert

Anstatt die GUI zu benutzen kann auch direkt in Python gearbeitet werden. Hier stehen auch weitere Operationrn wie der iterierte FGSM-Angriff und die Generation mehrer Täuschender Bilder auf einmal zu Verfügung. Zur Nutzung wird in dem heruntergeladenen Ordner über den Kommandozeilenbefehl Python eine Pythonkonsole geöffnet. Nun muss die entsprechende Klasse importiert werden (from sticker import StickerGenerator bzw. from fgsm inmport FGSM). Anschließend erstellt man mit Instanz = Klassenname(args) eine Instanz der entsprechenden Klasse und führt die gewünschte Methode mit Instanz.Methodenname(args) durch. "args" ist hier ein Platzhalter für die jeweiligen Argumente.

###### Modul fgsm:

FGSM(model=None, cuda=True): Erstellt eine Instanz der FGSM-Klasse, die als Basis für alls FGSM-Angriffe dient. FÜr model kann ein Pytorch-model angegeben werden, welches dann als White Box verwendet wird. Ohne Angabe wird die von uns trainierte White Box verwendet. Cuda bestimmt, ob Cuda genutzt werden soll.

Instanzmethoden:

preview_im(im_url): Für eine gültige Png-Bild-Url werde die fünf Klassen (mit zugehörigen numerischen Labels) mit der höchsten Konfidenz der Black Box ausgegeben.

simple_attack(im_url, save_url): Führt eine iterierte FGSM-Attack ausgehend vom Bild in im_url auf die vielversprechendste Klasse aus und speichert das Ergebnis bei Erfolg in save_url. Eine genauere Parametrisierung des Angriffes erfolgt über das Dictionary FGSM_SPECS in config.py, genau wie für die nächste Methode.

attack_on_label(im_url, save_url, target_label): Wie simple_attack, das anzugreifende numerische Label wird jedoch selbst ausgewählt.

simple_batch_attack(im_folder, save_folder, title=""): Führt simple_attack für alle Bilder in im_folder durch und speichert jedes Bild unter seinem Namen mit anghängtem title in save_folder.

batch_attack_on_label(im_folder,save_folder, target_label, title=""): Wie simple_batch_attack, nur mit selbstgewähltem Label


reload_model(model): Lädt eine neue Whitebox. Das zurücksetzen der Whitebox kann hilfreich sein, wenn diese durch zu viele Korrekturen in einer nicht repräsentativen Region nur noch verzerrte Resultate liefert.

###### Modul sticker:

	Die Klasse StickerGenerator kann mit den Werten pixelsize und fringe initialisiert werden (standardeinstellungen: 3, 17), die die Größe der Pixelblöcke, und deren Abstand zum Rand auf beiden Seiten bestimmen. 
	Dabei sollte die Bildgröße Minus 2 Mal fringe durch Pixelsize teilbar sein.
		Mit der (Instanz)Methode sticker_batch können nun Sticker generiert werden, die in dem Subverzeichnis für die entsprechende Klasse gespeichert werden. Mit dem optionalen Argument 	
		title (ohne Angabe: "") kann ein Titel für den Stickerbatch bestimmt werden. Für die einzlnen Sticker werden der Pixel_threshold und die Konfidenz an den Namen anghängt. 
		Pixel_threshold (ohne Angabe 0.01)  gibt vor, wie viel Konfidenzgewinn ein Pixel bringen muss um in den Sticker aufgenommen zu werden.
		Es werden diejenigen Sticker gespeichert, die auf schwarzem Hintergrund eine Konfidenz von mehr als save_threshold (ohne Angabe 0.9) für die zugehörige Klasse generieren. Je
		nach gewählten Parametern kann diese Methode einige Zeit in Anspruch nehmen, da für jeden Pixelblock aufgrund der Maximalanzahl von Anfragen an die Blackbox mindestens eine 
		Sekunde benötigt wird.

		Mit der Methode sticker_attack kann nun ein Sticker auf ein Bild geklebt und dieses gespeichert werden.
		Als input erhält die Methode die Url des Bildes und eine Ziel-Url zum Abspeichern als strings (in Anführungszeichen). Optional können eine bestimmte Zielklasse ("label", als Zahl codiert, siehe unten), 
		oder ein bestimmter Sticker als Url ("sticker_url") angegeben werden. Durch setzen des optionalen Argumentes "mode" auf "transparent" kann ein Sticker zudem transparent anstattfüllend aufgeklebt werden.


Die Parameter für die FGSM-Attacke befinden sich wie bereits erwähnt im python-dictionary FGSM_SPECS, welches in config.py 	 gespeichert ist. Im folgenden die Default-Einstellungen und eine Erklärung der Parameter:

	"mode": "l_inf" Gibt die Form der Projektion nach jeder Iteration an. "l_inf": Maximumsnorm "l_2": Euklidnorm "simple": keine Projektion. 

	"bound": 10  Gibt den Maximalwert der gewählten Norm an, auf den der Abstand dann zurück projiziert wird. Für mode="l_2" empfiehlt sich ein wert um 1000. 

	"magnitude": 1 Gibt das alpha/die Schrittgröße für die FGSM-Iteration an.

	"max_fgsm_iterations": 25 Gibt die Maximale Schrittanzahl in der FGSM-Iteration an. 

	"target_threshold": 0.99 Schwellenwert für die Konfidenz. Wenn dieser während der FGSM-Iteration überschritten wird, unterbricht diese und prüft den Abstand zwischen Black- und Whitebox und trainiert dann entweder die Whitebox nach, oder beendet bei Erfolg die Schleife.

	"fgsm_restart": "last" Bestimmt, ob nach dem adjustieren der Whitebox von vorne ("original") oder mit dem Endergebnis der letzten Iteration ("last") begonnen werden soll.

	"restart_max_amount": 10 Bestimmt die Maximale Anzahl an Versuchen, die Whitebox zu adjustieren.

	"restart_accuracy_bound": 0.0001 Beträgt der Quadrierte Abstand zwischen Vorhersage von White- und Blackbox am ende einer FGSM-Iterarion weniger als diese Schwelle, wird die Iteration abgebrochen.

	"retrain_mode": "last" Bestimmt, ob nur für die letzte FGSM-Iteration adjustiert werden soll ("last"), oder ob auch die Ergebnisse der vorigen Iterationen einbezogen werden sollen ("full").

	"retrain_lr": 0.00001 Bestimmt die Lerngeschwindigkeit beim Adjustieren. Niedrige werte verlangsamen die Anpassung, hohe führen zu lokaler Überanpassung oder sogar zu schlechterer Konvergenz.

	"retrain_max_gradient_steps": 10 Bestimmt die Anzahl der Optimierungsschritte beim Adjustieren. Effekte ählich wie bei der Lerngeschwindigkeit, hohe Werte verschlechtern die Konvergenz hier nicht, benötigen aber Zeit.

	"retrain_threshold": 0.0001 Bestimmt den Schwellenwert für den Quadratischen Abstand zwischen der Vorhersage von Black- und Whitebox, an dem das Adjustieren abgebrochen wird.

	"always_save": True Bestimmt, ob Endergebnisse, deren Konfidenz unter target_threshold liegt trotzdem gespeichert werden sollen.
	"print": True Bestimmt, ob Informationen, wie die Konfidenzen bei Zwischenschritten, ausgegeben werden sollen.

		
Weitere Config-Parameter:

	URL = 'https://phinau.de/trasi' , KEY = 'ut6ohb7ZahV9tahjeikoo1eeFaev1aef' (Abfrage der Blackbox)
	STICKER_DIRECTORY = "Quickstick" Superverzeichnis für das Abspeichern der Sticker.
	LABEL_AMOUNT = 43 Anzahl der Klassen
	IMAGE_SIZE = 64 Größe der Bilder in Pixeln
	WHITEBOX_DIRECTORY = "Models/ResNet.pt" Speicherort des Modells für die White Box

Liste der Klassen und der zugordneten numerischen Label: (CLASSNAMEDICT)

	 'Zulässige Höchstgeschwindigkeit (20)': 0,
	 'Zulässige Höchstgeschwindigkeit (30)': 1,
	 'Zulässige Höchstgeschwindigkeit (50)': 2,
	 'Zulässige Höchstgeschwindigkeit (60)': 3,
	 'Zulässige Höchstgeschwindigkeit (70)': 4,
	 'Zulässige Höchstgeschwindigkeit (80)': 5,
	 'Ende der Geschwindigkeitsbegrenzung (80)': 6,
	 'Zulässige Höchstgeschwindigkeit (100)': 7,
	 'Zulässige Höchstgeschwindigkeit (120)': 8,
	 'Überholverbot für Kraftfahrzeuge aller Art': 9,
	 'Überholverbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t': 10,
	 'Einmalige Vorfahrt': 11,
	 'Vorfahrt': 12,
	 'Vorfahrt gewähren': 13,
	 'Stoppschild': 14,
	 'Verbot für Fahrzeuge aller Art': 15,
	 'Verbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse von 3,5t': 16,
	 'Verbot der Einfahrt': 17,
	 'Gefahrenstelle': 18,
	 'Kurve (links)': 19,
	 'Kurve (rechts)': 20,
	 'Doppelkurve (zunächst links)': 21,
	 'Unebene Fahrbahn': 22,
	 'Schleudergefahr bei Nässe oder Schmutz': 23,
	 'Fahrbahnverengung (rechts)': 24,
	 'Baustelle': 25,
	 'Lichtzeichenanlage': 26,
	 'Fußgänger': 27,
	 'Kinder': 28,
	 'Fahrradfahrer': 29,
	 'Schnee- oder Eisglätte': 30,
	 'Wildwechsel': 31,
	 'Ende aller Streckenverbote': 32,
	 'Ausschließlich rechts': 33,
	 'Ausschließlich links': 34,
	 'Ausschließlich geradeaus': 35,
	 'Ausschließlich geradeaus oder rechts': 36,
	 'Ausschließlich geradeaus oder links': 37,
	 'Rechts vorbei': 38,
	 'Links vorbei': 39,
	 'Kreisverkehr': 40,
	 'Ende des Überholverbotes für Kraftfahrzeuge aller Art': 41,
	 'Ende des Überholverbotes für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t': 42
