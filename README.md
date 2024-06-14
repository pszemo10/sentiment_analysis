# Analiza sentymentu recenzji

Celem tego projektu jest prezentacja modelu do analizy sentymentu recenzji na podstawie recenzji produktów z platformy Amazon.
Wszystkie pliki źródłowe znajdują się w folderze ```src``` natomiast dane powinny znaleźć się w folderze ```data```. Dane te zawierają tekst recenzji oraz ocenę pozytywną lub negatywną. Wytrenowany model znajdzie się w folderze ```model```, na podstawie tekstu recenzji przewiduje on sentyment pozytywny lub negatywny.

## Wymagania

Potrzebne zależności znajdują się w pliku requirements.txt. Wpisane są tam ich wersje, których używałem podczas robienia projektu.
By je zainstalować należy wykonać polecenie 
``` 
pip install -r requirements.txt
```

## Dane

By pobrać dane w głównym katalogu projektu należy utworzyć folder ```data``` poleceniem
```
mkdir data
```
Dane dostępne są na platformie Kaggle https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
Po ich pobraniu należy je rozpakować do folderu ```data```. Po rozpakowaniu w folderze tym powinny znaleźć się pliki ```train.ft.txt``` oraz ```test.ft.txt```.

## Trenowanie

By wytrenować model w głównym katalogu projektu należy utworzyć folder ```model``` poleceniem
```
mkdir model
```
W tym folderze zapisywany będzie model i tokenizer.
By mieć pewność, że nowo trenowany model się zapisze folder ten nie powinien zawierać poprzednio wytrenowanego modelu.
By wytrenować model przechodzimy do folderu ```src```
```
cd src
```
Trenowanie uruchamiamy poleceniem
```
python3 Training.py
```
Domyślnie model trenowany jest na danych z pliku ```train.ft.txt``` zawierającym 3 600 000 danych. Ze względu na ograniczenia sprzętowe trenowałem model na pierwszych 1 000 000 danych co zajmowało około 5 godzin.
By przyspieszyć ten proces można ograniczyć trenowanie do jednej epoki zmieniając parametr ```epochs``` w linii 33. na 1 lub zmniejszyć ilość wczytywanych danych. Nawet dla 10000 można uzyskać accuracy na poziomie 85%. Liczbę linii pliku używanych do treningu lub ewaluacji można zmienić odkomentowując linię 19. w pliku Dataloader.py.

## Ewaluacja

Testowanie modelu odbywa się za pomocą wykonania w tym samym folderze polecenia
```
python3 Testing.py
```
Testowany model oraz tokenizer wczytywane są z folderu ```model``` dlatego przed ewaluacją powinny znajdować się tam folder ```model_weights.h5py``` z wytrenowanym modelem oraz plik ```tokenizer.json``` z tokenizerem. 
