# ASLCaptureService dokumentacija

## Svrha datoteke

Datoteka `asl_capture_service.py` implementira servis za ručno i poluautomatsko prikupljanje vlastitog ASL skupa podataka pomoću web-kamere. Glavna ideja je da korisnik odabere trenutno slovo, pokaže odgovarajući znak rukom prema kameri, a program zatim sprema:

- cijeli kadar kamere
- izrezanu regiju ruke (`crop`)

Ovaj servis je koristan za izradu personaliziranog dataseta koji bolje odgovara stvarnim uvjetima rada nego generički skupovi podataka. Time se kasnije može poboljšati treniranje i evaluacija modela za prepoznavanje gesti ruke.

## Gdje se koristi

`ASLCaptureService` se tipično poziva iz glavne ulazne skripte `mainMedia.py` kroz naredbu:

```powershell
python mainMedia.py capture-asl
```

Taj servis ne trenira model i ne radi klasifikaciju slova. Njegova odgovornost je isključivo:

- otvoriti kameru
- prepoznati ruku pomoću MediaPipe-a
- prikazati vizualizaciju ruke i overlay informacija
- spremati slike za odabrano slovo

## Ovisnosti

`asl_capture_service.py` koristi sljedeće biblioteke i module:

- `ctypes`
  - dohvat fizičkog stanja tipke `Space` na Windowsu
- `cv2`
  - pristup kameri
  - iscrtavanje teksta
  - prikaz prozora
  - enkodiranje i osnovna obrada slike
- `numpy`
  - rad sa slikama kao matricama
- `dataclasses`
  - definicija pomoćne strukture `SaveResult`
- `pathlib.Path`
  - rad s direktorijima i putanjama
- `string.ascii_uppercase`
  - generiranje popisa slova `A-Z`
- `time.perf_counter`
  - precizno mjerenje vremena za rate-limited spremanje
- `AppConfig`
  - centralna konfiguracija projekta
- `MediaPipeFeatureExtractor`
  - obrada framea, detekcija ruke, generiranje anotiranog prikaza i obojane maske

## Opći tijek rada

Servis radi po sljedećem principu:

1. Inicijalizira se objekt `ASLCaptureService`.
2. Otvori se kamera.
3. Kamera kontinuirano šalje frameove.
4. Svaki frame se horizontalno zrcali radi prirodnijeg prikaza korisniku.
5. `MediaPipeFeatureExtractor` analizira frame i vraća informacije o ruci.
6. Gradi se prikaz s dva panela:
   - lijevo: originalna kamera s landmark točkama i anotacijama
   - desno: stilizirana obojana maska ruke
7. Iznad i ispod slike dodaju se crni informacijski barovi.
8. Korisnik tipkama mijenja slovo ili način spremanja.
9. Spremljene slike odlaze u direktorij za trenutno slovo.

## Kontrole tipkovnicom

Servis podržava sljedeće tipke:

- `Space`
  - dok se drži, radi kontinuirano spremanje ograničeno aktivnim capture rateom
- `B`
  - uključivanje ili isključivanje burst moda
- `Arrow Up`
  - prelazak na sljedeće slovo
- `Arrow Down`
  - povratak na prethodno slovo
- `Arrow Left`
  - smanjuje capture rate za `1 fps`
- `Arrow Right`
  - povećava capture rate za `1 fps`
- `W`
  - alternativa za `Arrow Up`
- `S`
  - alternativa za `Arrow Down`
- `A`
  - alternativa za `Arrow Left`
- `D`
  - alternativa za `Arrow Right`
- `Q`
  - izlaz iz aplikacije
- `Esc`
  - izlaz iz aplikacije

Capture rate ima granice:

- minimum `5 fps`
- maksimum `30 fps`

Zadana vrijednost pri pokretanju je `10 fps`.

## Struktura spremljenih podataka

Za svako slovo kreira se zaseban direktorij. Struktura je:

```text
ASL/
  captured/
    A/
      crops/
      frames/
    B/
      crops/
      frames/
    ...
    Z/
      crops/
      frames/
```

Primjer naziva datoteka:

```text
A_00001_crop.png
A_00001_frame.png
```

To znači:

- `crop`
  - izrezana regija ruke
- `frame`
  - cijeli kadar kamere u trenutku spremanja

Takva podjela je korisna jer `crop` može služiti za treniranje klasifikatora, dok `frame` zadržava širi kontekst za kasniju provjeru kvalitete dataseta.

## Konstante za tipke

### `UP_KEYS`

```python
UP_KEYS = {2490368, 82, ord("w"), ord("W")}
```

Ovaj skup sadrži kodove tipki koje znače "idi na sljedeće slovo". Uključene su:

- stvarne strelice gore koje `cv2.waitKeyEx()` vrati na pojedinim sustavima
- `W` i `w` kao rezervna varijanta

Razlog za ovo je praktičan: OpenCV ponekad različito prijavljuje specijalne tipke ovisno o sustavu, drajveru i okruženju.

### `DOWN_KEYS`

```python
DOWN_KEYS = {2621440, 84, ord("s"), ord("S")}
```

Ista logika kao iznad, ali za kretanje prema prethodnom slovu.

### `LEFT_KEYS`

```python
LEFT_KEYS = {2424832, 81, ord("a"), ord("A")}
```

Skup tipki za smanjivanje capture ratea.

### `RIGHT_KEYS`

```python
RIGHT_KEYS = {2555904, 83, ord("d"), ord("D")}
```

Skup tipki za povećavanje capture ratea.

## Pomoćna dataklasa `SaveResult`

```python
@dataclass(slots=True)
class SaveResult:
    saved: bool
    message: str
    path: Path | None = None
```

### Svrha

`SaveResult` je mali objekt koji opisuje ishod pokušaja spremanja slike.

### Polja

- `saved`
  - `True` ako je spremanje uspjelo
  - `False` ako nije
- `message`
  - tekstualna poruka za korisnika
  - prikazuje se u overlayu kao status
- `path`
  - putanja do spremljenog cropa ako je spremanje uspjelo
  - inače `None`

### Zašto je koristan

Bez ovog objekta metoda za spremanje bi morala vraćati više odvojenih vrijednosti ili bi morala ručno mijenjati stanje izvan sebe. `SaveResult` daje čist i jasan način prijenosa rezultata.

## Klasa `ASLCaptureService`

Ova klasa predstavlja središnji servis za capture logiku.

---

## Metoda `__init__(self, config: AppConfig) -> None`

### Uloga

Konstruktor priprema osnovne parametre i stanje servisa.

### Što postavlja

#### `self.config`

Referenca na konfiguracijski objekt aplikacije. Kroz njega servis dobiva, između ostalog, putanju za spremanje ASL capture podataka.

#### `self.letters`

```python
self.letters = list(ascii_uppercase)
```

Pretvara `A-Z` u listu:

```python
["A", "B", "C", ..., "Z"]
```

Ta lista se koristi za:

- odabir trenutnog slova
- generiranje direktorija po slovima
- prikaz sažetka po slovima

#### `self.blur_threshold`

```python
self.blur_threshold = 85.0
```

Minimalni prag oštrine slike ispod kojeg se kadar neće spremiti. Oštrina se procjenjuje varijancom Laplace operatora. Ako je vrijednost preniska, slika se smatra mutnom.

#### `self.min_capture_fps`

```python
self.min_capture_fps = 5.0
```

Najmanja dozvoljena brzina spremanja.

#### `self.max_capture_fps`

```python
self.max_capture_fps = 30.0
```

Najveća dozvoljena brzina spremanja.

#### `self.capture_fps`

```python
self.capture_fps = 10.0
```

Aktivni broj spremanja u sekundi. Ovu vrijednost koriste:

- držanje `Space`
- `burst mode`

#### `self.capture_interval_seconds`

```python
self.capture_interval_seconds = 1.0 / self.capture_fps
```

Pretvara aktivni capture rate u vremenski interval. Ako je `capture_fps = 10`, interval je `0.1 s`. To znači da servis neće spremiti više od jedne slike svakih 100 ms.

### Sažetak

Konstruktor ne otvara kameru i ne radi obradu. On samo priprema parametre potrebne za rad.

---

## Metoda `run(self, camera_index: int = 0) -> None`

### Uloga

Ovo je glavna radna petlja servisa. Ona otvara kameru, čita frameove, obrađuje ih i reagira na korisnički unos.

### Parametar

- `camera_index`
  - indeks kamere koji se predaje `cv2.VideoCapture`
  - zadano `0`, što je obično primarna web-kamera

### Koraci metode

#### 1. Otvaranje kamere

```python
capture = cv2.VideoCapture(camera_index)
```

Kreira OpenCV objekt za čitanje sa zadane kamere.

#### 2. Provjera dostupnosti kamere

```python
if not capture.isOpened():
    raise RuntimeError(...)
```

Ako kamera nije dostupna, odmah se prekida s jasnom porukom greške.

#### 3. Postavljanje rezolucije

```python
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
```

Ovdje se traži rezolucija `960x540`. Kamera to ne mora nužno točno ispoštovati, ali OpenCV pokušava postaviti tražene dimenzije.

#### 4. Inicijalizacija lokalnog stanja

Postavljaju se varijable:

- `current_index = 0`
  - početno slovo je `A`
- `status_text`
  - početna poruka za korisnika
- `burst_mode = False`
  - burst je na početku isključen
- `last_capture_save = 0.0`
  - vrijeme zadnjeg spremanja koje podliježe rate limiteru
- `letter_counts = self._load_letter_counts()`
  - učitava postojeći broj slika po slovima

Time servis nastavlja brojanje i kad se aplikacija ponovno pokrene.

#### 5. Korištenje `MediaPipeFeatureExtractor`

```python
with MediaPipeFeatureExtractor(static_image_mode=False) as extractor:
```

Extractor se otvara kao context manager. To znači da će se resursi MediaPipe-a korektno zatvoriti nakon izlaska iz bloka.

`static_image_mode=False` znači da se extractor koristi za kontinuirani video stream, a ne za potpuno nepovezane slike.

#### 6. Glavna beskonačna petlja

```python
while True:
```

U svakoj iteraciji petlje:

- čita se frame
- analizira se ruka
- gradi se preview
- provjerava se treba li spremati kadar
- crta se overlay
- čeka se tipka

#### 7. Čitanje framea

```python
success, frame = capture.read()
```

Ako `success` nije `True`, baca se greška jer se kadar nije mogao pročitati.

#### 8. Horizontalno zrcaljenje

```python
frame = cv2.flip(frame, 1)
```

Zrcaljenje se radi zato što je korisnicima prirodnije gledati sebe kao u ogledalu. Bez toga pomicanje lijevo-desno djeluje kontraintuitivno.

#### 9. Analiza framea

```python
detection = extractor.analyze_frame(frame)
```

Rezultat sadrži barem sljedeće relevantne stvari:

- `annotated_frame`
  - originalni frame s nacrtanim landmark točkama i linijama
- `color_mask`
  - stilizirani prikaz ruke u bojama
- `bbox`
  - bounding box ruke
- `detected`
  - informaciju je li ruka pronađena
- `finger_states`
  - stanje svakog prsta

#### 10. Sastavljanje prikaza

```python
preview = self._build_preview(detection.annotated_frame, detection.color_mask)
```

Lijevi i desni panel spajaju se u jednu sliku.

#### 11. Dohvat trenutnog slova i broja spremljenih slika

```python
current_letter = self.letters[current_index]
saved_count = letter_counts[current_letter]
```

#### 12. Logika držanja `Space`

Na Windows sustavu servis provjerava je li `Space` trenutno fizički pritisnut:

```python
if self._is_space_pressed() and (now - last_capture_save) >= self.capture_interval_seconds:
```

To znači:

- korisnik može držati `Space`
- program će spremati kontinuirano
- ali neće prekoračiti aktivni `capture_fps`

Drugim riječima, držanje `Space` radi kao ručno aktivirani mini-burst.

#### 13. Burst logika

Ako je `burst_mode` uključen, koristi se ista rate-limit logika:

```python
if burst_mode and (now - last_capture_save) >= self.capture_interval_seconds:
```

Ako je proteklo dovoljno vremena:

- pokušava se spremiti novi uzorak
- ako spremanje uspije, ažurira se `last_capture_save`
- status se osvježava `[BURST] ...`

Važno: ni `Space-hold` ni `burst` ne jamče točno zadani broj spremanja u sekundi. To je ciljano ponašanje, ali stvarni broj ovisi o:

- brzini kamere
- brzini MediaPipe obrade
- brzini kodiranja PNG-a
- brzini diska

#### 14. Dodavanje overlaya

```python
preview = self._draw_overlay(...)
```

Na dobivenu spojenu sliku dodaju se:

- gornji crni bar
- donji crni bar
- statusne informacije
- aktivni capture rate
- sažetak po slovima

#### 15. Prikaz prozora

```python
cv2.imshow("ASL Capture", preview)
```

#### 16. Čitanje tipke

```python
key = cv2.waitKeyEx(1)
```

Koristi se `waitKeyEx` umjesto `waitKey` jer bolje vraća kodove specijalnih tipki poput strelica.

#### 17. Obrada ulaza

Podržani slučajevi:

- `Q`, `q`, `Esc`
  - izlaz
- `UP_KEYS`
  - povećanje indeksa slova
- `DOWN_KEYS`
  - smanjenje indeksa slova
- `LEFT_KEYS`
  - smanjenje `capture_fps`
- `RIGHT_KEYS`
  - povećanje `capture_fps`
- `Space`
  - fallback jednokratno spremanje samo ako sustav ne podržava provjeru fizičkog stanja tipke
- `B`, `b`
  - uključivanje/isključivanje burst moda

### `try/finally` dio

Bez obzira na način izlaska iz petlje, izvršit će se:

```python
capture.release()
cv2.destroyAllWindows()
```

To je bitno za:

- oslobađanje kamere
- zatvaranje OpenCV prozora
- sprečavanje zaključavanja hardverskog resursa

---

## Metoda `_save_current_sample(...) -> SaveResult`

### Uloga

Pokušava spremiti jedan uzorak za trenutno slovo.

### Parametri

- `letter`
  - trenutno aktivno slovo
- `frame`
  - cijeli frame kamere
- `bbox`
  - bounding box ruke
- `counts`
  - rječnik s brojem spremljenih slika po slovima

### Logika metode

#### 1. Provjera `bbox`

```python
if bbox is None:
    return SaveResult(...)
```

Ako MediaPipe nije dao dovoljno pouzdan bounding box ruke, spremanje se preskače. Time se izbjegava spremanje besmislenih kadrova bez jasne ruke.

#### 2. Priprema direktorija

Metoda formira putanje:

- `target_dir`
- `crops_dir`
- `frames_dir`

Zatim osigurava da ti direktoriji postoje:

```python
mkdir(parents=True, exist_ok=True)
```

#### 3. Generiranje naziva datoteka

```python
image_index = counts[letter] + 1
```

Time se broj za novo slovo nastavlja od već postojećeg stanja.

#### 4. Proširenje bounding boxa

```python
x1, y1, x2, y2 = self._expand_bbox(bbox, frame.shape)
```

Originalni bounding box ruke se malo proširi kako crop ne bi bio pretijesan. Ovo je važno jer:

- vrhovi prstiju inače mogu biti odsječeni
- uz rubove ruke treba ostati malo konteksta

#### 5. Izrezivanje ruke

```python
crop = frame[y1 : y2 + 1, x1 : x2 + 1]
```

Ako je crop prazan, spremanje se prekida.

#### 6. Procjena mutnoće

```python
blur_score = self._blur_score(crop)
if blur_score < self.blur_threshold:
    return SaveResult(...)
```

Ako je crop previše mutan, odbacuje se. Time se kvaliteta dataseta održava višom.

#### 7. PNG enkodiranje

```python
crop_ok, crop_encoded = cv2.imencode(".png", crop)
frame_ok, frame_encoded = cv2.imencode(".png", frame)
```

Prije spremanja datoteka, slike se eksplicitno enkodiraju u PNG. To omogućuje jednostavno zapisivanje bajtova preko `Path.write_bytes(...)`.

#### 8. Upis na disk

Ako je enkodiranje uspjelo:

- zapisuje se crop
- zapisuje se cijeli frame
- brojač za slovo se povećava

#### 9. Povrat rezultata

Metoda vraća `SaveResult` s porukom tipa:

```text
Spremljeno A: crop + frame #00001 | blur=123.4
```

### Zašto je metoda važna

Ova metoda je centralna za kvalitetu dataseta. Ona osigurava da se:

- ne sprema kadar bez ruke
- ne sprema prazan crop
- ne sprema očito mutna slika

---

## Metoda `_build_preview(self, annotated_frame, color_mask) -> np.ndarray`

### Uloga

Sastavlja završni vizualni pregled od dva panela.

### Parametri

- `annotated_frame`
  - lijevi panel s prikazom kamere i landmark točkama
- `color_mask`
  - desni panel s obojanom maskom ruke

### Implementacija

```python
left = annotated_frame.copy()
right = color_mask.copy()
return np.hstack([left, right])
```

### Objašnjenje

- `copy()` sprječava nenamjerne izmjene izvornog niza
- `np.hstack(...)` spaja slike horizontalno, jednu pored druge

Rezultat je slika dvostruke širine:

- lijevo kamera
- desno vizualizacija ruke

---

## Metoda `_draw_overlay(...) -> np.ndarray`

### Uloga

Dodaje informacijske barove i tekst preko pripremljenog previewa.

### Parametri

- `image`
  - već sastavljen preview
- `current_letter`
  - trenutno aktivno slovo
- `saved_count`
  - broj slika spremljenih za trenutno slovo
- `letter_counts`
  - broj slika za sva slova
- `status_text`
  - poruka o zadnjoj akciji
- `hand_detected`
  - je li ruka trenutno detektirana
- `finger_states`
  - stanje prstiju
- `burst_mode`
  - informacija je li burst aktivan

### Koraci metode

#### 1. Kreiranje novog platna

```python
top_bar_height = 100
bottom_bar_height = 100
canvas = np.zeros((height + top_bar_height + bottom_bar_height, width, 3), dtype=np.uint8)
```

Ovdje se kreira nova crna slika koja je viša od previewa za dodatnih 200 piksela:

- 100 px gore
- 100 px dolje

#### 2. Umetanje preview slike u sredinu

```python
canvas[top_bar_height : top_bar_height + height, :] = image
```

Tako preview ostaje netaknut u sredini, a tekst ne prekriva samu sliku ruke.

#### 3. Računanje ukupnog broja spremljenih slika

```python
total_saved = sum(letter_counts.values())
```

#### 4. Definiranje gornjih linija

Gornji blok prikazuje:

- trenutno slovo
- broj spremljenih slika za trenutno slovo
- ukupan broj spremljenih slika, aktivni rate i burst status
- status detekcije ruke i blur threshold

Svaka linija ima:

- labelu
- vrijednost
- boju

To omogućuje fleksibilno i uredno crtanje.

#### 5. Definiranje donjih linija

Donji blok prikazuje:

- stanje prstiju
- status zadnje akcije
- keybinds

Posebno je korisno što se keybinds crtaju drugom bojom, pa se lakše odvajaju od statusnih informacija.

#### 6. Crtanje gornjih linija

Za svaku liniju:

- prvo se crta labela
- zatim se računa širina labele
- onda se odmah iza nje crta vrijednost

To se radi pomoću `_text_width(...)`.

#### 7. Crtanje sažetka A-Z

```python
summary_lines = self._build_letter_summary_lines(letter_counts)
```

Sažetak se crta na desnoj polovici gornjeg bara i prikazuje broj spremljenih uzoraka za sva slova.

#### 8. Crtanje donjih linija

Ista logika kao kod gornjih linija, samo se crta unutar donjeg bara.

### Vizualna uloga metode

Ova metoda rješava jedan važan problem: informacije ne prekrivaju samu ruku. To je posebno bitno kad se korisnik mora vizualno oslanjati na desni panel pri pozicioniranju prstiju.

---

## Metoda `_format_finger_states(self, finger_states: dict[str, str]) -> str`

### Uloga

Pretvara rječnik stanja prstiju u čitljiv tekst.

### Primjer ulaza

```python
{
    "palac": "ispruzen",
    "kaziprst": "savijen",
    "srednji": "polu-savijen",
}
```

### Primjer izlaza

```text
Prsti: palac=ispruzen | kaziprst=savijen | srednji=polu-savijen
```

### Zašto postoji definirani redoslijed

Metoda koristi:

```python
ordered_names = ["palac", "kaziprst", "srednji", "prstenjak", "mali"]
```

To osigurava da se prikaz uvijek pojavljuje istim redom, bez obzira na interni poredak ključeva u rječniku.

Ako nema podataka, vraća:

```text
Prsti: nema podataka
```

---

## Metoda `_load_letter_counts(self) -> dict[str, int]`

### Uloga

Prilikom pokretanja aplikacije prebrojava već spremljene slike po slovima.

### Zašto je važna

Bez ove metode bi svako novo pokretanje aplikacije krenulo ispočetka od `1`, što bi:

- prebrisalo stare datoteke
- pokvarilo kontinuitet brojeva

### Kako radi

Za svako slovo:

1. Formira putanju do `crops` direktorija.
2. Ako direktorij ne postoji, broj se postavlja na `0`.
3. Ako postoji, broje se datoteke s ekstenzijama:
   - `.png`
   - `.jpg`
   - `.jpeg`

Iako servis trenutno sprema PNG, metoda je tolerantna i na druge formate.

---

## Metoda `_build_letter_summary_lines(self, letter_counts: dict[str, int]) -> list[str]`

### Uloga

Pretvara broj uzoraka po slovima u tri pregledne tekstualne linije za overlay.

### Logika grupiranja

Slova se grupiraju ovako:

- `A-I`
- `J-R`
- `S-Z`

To je kompromis između:

- čitljivosti
- raspoloživog prostora
- potrebe da se svih 26 slova vide odjednom

### Primjer izlaza

```text
A:12 | B:4 | C:0 | D:8 | E:3 | F:7 | G:2 | H:1 | I:0
J:0 | K:0 | L:5 | M:0 | N:0 | O:0 | P:0 | Q:0 | R:0
S:0 | T:0 | U:0 | V:0 | W:0 | X:0 | Y:0 | Z:0
```

---

## Metoda `_blur_score(self, image: np.ndarray) -> float`

### Uloga

Procjenjuje oštrinu slike.

### Implementacija

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
return float(cv2.Laplacian(gray, cv2.CV_64F).var())
```

### Objašnjenje

Metoda radi ovako:

1. Slika se pretvara u grayscale.
2. Računa se Laplace operator.
3. Uzima se varijanca rezultata.

### Tumačenje rezultata

- veća vrijednost
  - više rubova i detalja
  - slika je oštrija
- manja vrijednost
  - manje detalja
  - slika je mutnija

### Zašto je korisna

Kod ručnog prikupljanja dataseta lako je nenamjerno spremiti:

- kadar u pokretu
- kadar bez fokusa
- kadar sa zamućenim prstima

Ova metoda služi kao jednostavan filter kvalitete.

---

## Metoda `_text_width(self, text: str, font_scale: float) -> int`

### Uloga

Računa širinu teksta u pikselima za zadani font i veličinu.

### Implementacija

```python
(width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
return width
```

### Zašto je potrebna

Overlay tekst se crta u formatu:

```text
Labela: vrijednost
```

Kako bi vrijednost počela točno iza labele, prvo treba znati koliko piksela labela zauzima.

Ova metoda pomaže u urednom poravnanju bez ručnog nagađanja x-koordinata.

---

## Metoda `_set_capture_fps(self, fps: float) -> None`

### Uloga

Postavlja novi capture rate i odmah ažurira pripadni vremenski interval.

### Logika

Vrijednost se steže unutar dozvoljenih granica:

- najmanje `5 fps`
- najviše `30 fps`

Nakon toga se računa:

```python
self.capture_interval_seconds = 1.0 / self.capture_fps
```

To osigurava da ostatak aplikacije uvijek koristi sinkronizirane vrijednosti.

---

## Metoda `_supports_key_state(self) -> bool`

### Uloga

Provjerava podržava li trenutno okruženje dohvat fizičkog stanja tipke preko Windows API-ja.

### Zašto postoji

`cv2.waitKeyEx()` dobro hvata pritiske tipki, ali za držanje `Space` nije idealan na svim sustavima. Zato se na Windowsu koristi `GetAsyncKeyState`, ali samo ako je dostupan.

---

## Metoda `_is_space_pressed(self) -> bool`

### Uloga

Provjerava je li `Space` trenutno fizički pritisnut.

### Implementacijska ideja

Ako postoji podrška za Windows `user32` API, metoda poziva:

```python
ctypes.windll.user32.GetAsyncKeyState(0x20)
```

Time se dobiva stvarno stanje tipke, što omogućuje da držanje `Space` radi kao kontinuirani capture, umjesto da ovisi samo o ponavljanju key eventova.

---

## Metoda `_expand_bbox(self, bbox, frame_shape) -> tuple[int, int, int, int]`

### Uloga

Proširuje bounding box ruke prije cropanja.

### Zašto je to potrebno

MediaPipe bounding box često tijesno obuhvati ruku. To može uzrokovati:

- odsječene vrhove prstiju
- pretijesan crop
- manje robusne uzorke za treniranje

### Logika metode

Iz ulaznog bboxa:

```python
x1, y1, x2, y2 = bbox
```

računa se padding:

```python
pad_x = max(int((x2 - x1) * 0.18), 18)
pad_y = max(int((y2 - y1) * 0.18), 18)
```

To znači:

- padding je 18% širine/visine bounding boxa
- ali nikad manji od 18 piksela

### Granice slike

Nakon proširenja, koordinate se ograniče na stvarne granice framea:

- ne smije ići lijevo ispod `0`
- ne smije ići gore ispod `0`
- ne smije ići desno preko širine
- ne smije ići dolje preko visine

To sprečava pogreške pri izrezivanju.

---

## Povezanost s `MediaPipeFeatureExtractor`

`ASLCaptureService` sam po sebi ne detektira ruku. On se oslanja na `MediaPipeFeatureExtractor`, koji mu daje:

- položaj ruke
- vizualne landmarke
- obojanu masku
- stanja prstiju

Drugim riječima:

- `ASLCaptureService` upravlja kamerom, UI-em i spremanjem
- `MediaPipeFeatureExtractor` upravlja analizom ruke

Ta podjela odgovornosti je dobra jer odvojeno drži:

- računalni vid
- korisnički interfejs
- logiku spremanja dataseta

## Prednosti trenutne implementacije

- jednostavno ručno prikupljanje podataka
- automatsko razdvajanje po slovima
- podrška za burst spremanje
- podrška za držanje `Space` uz rate limiter
- podesiv capture rate `5-30 fps`
- provjera mutnoće slike
- vizualni feedback u stvarnom vremenu
- nastavak brojanja nakon ponovnog pokretanja aplikacije
- čuvanje i cropa i cijelog framea

## Ograničenja trenutne implementacije

- nema automatskog prepoznavanja koje slovo korisnik pokazuje
- koristi jedan globalni `blur_threshold`, ne adaptira se po osvjetljenju
- koristi jednu kameru i jedan indeks kamere
- `GetAsyncKeyState` dio je Windows-specifičan
- ne zapisuje dodatne metapodatke u CSV ili JSON
- ne radi automatsku provjeru stabilnosti poze prije spremanja
- oslanja se na kvalitetu MediaPipe detekcije u realnom vremenu

## Preporuke za buduću nadogradnju

Za kasniji razvoj diplomske aplikacije korisne nadogradnje bile bi:

- spremanje metapodataka u CSV
  - slovo
  - timestamp
  - blur score
  - putanja do cropa
  - putanja do framea
- automatsko spremanje tek kad je poza stabilna nekoliko frameova
- opcija za brisanje zadnje spremljene slike
- opcija za promjenu `blur_threshold` kroz UI
- opcija za promjenu početnog `capture_fps` kroz CLI argument
- prikaz FPS-a i performansi u overlayu
- spremanje dodatne maske ili landmark koordinata uz sliku

## Kratki zaključak

`ASLCaptureService` je specijalizirani servis za izgradnju vlastitog ASL image dataseta pomoću kamere. Njegova glavna vrijednost nije u klasifikaciji nego u kontroliranom i organiziranom prikupljanju podataka. Kombinacija:

- MediaPipe detekcije
- cropanja ruke
- provjere mutnoće
- spremanja po slovima
- rate-limitiranog `Space` capturea
- podesivog burst/capture ratea
- preglednog vizualnog interfejsa

čini ga dobrim temeljem za daljnji razvoj sustava za prepoznavanje znakova i gesta ruke.

## Dodatna napomena

Font overlay teksta je postavljen na `Calibri` ili `Arial` kada je dostupan na sustavu, radi bolje čitkosti informacija u prozoru aplikacije.
