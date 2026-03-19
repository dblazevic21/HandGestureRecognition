# MediaPipe projekt za prepoznavanje gesti ruke

Ovaj projekt prolazi kroz sve slike iz zadanih dataset foldera, koristi `MediaPipe Hands` za detekciju ruke i izvlacenje landmarksa, te zatim trenira klasifikator nad tim znacajkama. Nakon toga mozes pokrenuti kameru i prepoznavati geste u realnom vremenu.

## Sto projekt trenutno radi

- skenira slike iz foldera
- `24000-900-300t`
- `L-thumbs`
- `L-thumbs/dataset`
- `leapGestRecog`
- `MediaPipe`
- `rock-paper-scissors`
- `rock-paper-scissors/HandGesture`
- `rock-paper-scissors/HandGesture/images`
- automatski uzima labelu iz zadnjeg nazivnika foldera
- deduplicira ocite duplikate, npr. u `leapGestRecog`
- trenira model na MediaPipe landmark znacajkama
- sprema model i sazetak treniranja u `artifacts/`
- pokrece webcam prikaz s dva panela:
  - lijevo normalna kamera s MediaPipe prikazom
  - desno crno-bijela maska gdje je ruka bijela, a pozadina crna

## Struktura

- `mainMedia.py`
- `mediapipe_app/dataset_scanner.py`
- `mediapipe_app/feature_extractor.py`
- `mediapipe_app/model_service.py`
- `mediapipe_app/webcam_service.py`

## Pokretanje

```powershell
cd "C:\Users\Korisnik\OneDrive\Radna površina\Diplomski\MediaPipe"
python -m pip install -r requirements.txt
python mainMedia.py scan
python mainMedia.py train --max-images-per-class 200 --progress-every 100
python mainMedia.py webcam
```

Za rucno snimanje vlastitih ASL slika preko kamere:

```powershell
python mainMedia.py capture-asl
```

Ponasanje capture moda:

- trenutno aktivno slovo pise dolje na ekranu
- `Space` sprema trenutnu sliku u `ASL\captured\<slovo>`
- strelica gore ide prema `Z`
- strelica dolje ide prema `A`
- ako folder za slovo ne postoji, automatski se kreira
- sprema se crop ruke, ne cijeli kadar

Za brzi test s manjim brojem slika po klasi:

```powershell
python mainMedia.py train --max-images-per-class 200 --progress-every 100
```

Za puni run nad svim slikama:

```powershell
python mainMedia.py train
```

Za treniranje i odmah kameru:

```powershell
python mainMedia.py full
```

## Artefakti

Nakon treniranja u `artifacts/` dobijes:

- `gesture_model.joblib`
- `training_summary.json`
- `dataset_summary.json`
- `training_class_summary.csv`
- `training_reason_summary.csv`
- `training_fallback_summary.csv`
- `training_diagnostics.csv`

## Napomena za temu rada

Ovo je dobar pocetni MediaPipe baseline za diplomski rad jer koristi isti izvor slika za treniranje i kasnije prepoznavanje, daje mjerljive rezultate i lako se kasnije usporedjuje s drugim pristupima.
