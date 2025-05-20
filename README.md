Ovaj projekt implementira biometrijsku zaštitu predložaka lica korištenjem fuzzy vault sustava. Umjesto da se slike ili izravni biometrijski podaci pohranjuju, koristi se kombinacija dubokog učenja i kriptografije kako bi se iz slike lica generirao siguran, neobnovljiv predložak otporan na krađu i zloupotrebu.

Projekt je namijenjen za demonstraciju zaštite biometrijskih podataka, evaluaciju sigurnosti fuzzy vault sustava i kao osnova za daljnja istraživanja ili razvoj.


Struktura projekta:
    
    folder /data - sastoji se od tri podfoldera
  
    /training - slike za treniranje modela
    /gallery - slike za registraciju korisnika
    /probe - slike za verifikaciju
  
    folder /weights
      /models - ovdje će InsightFace automatski preuzeti model buffalo_l
    
    folder /src - sadrži glavni kod kojeg pokrećemo
  
    config.ini - datoteka u kojoj postavljamo parametre

  Ključne funkcionalnosti:

    Ekstrakcija značajki lica pomoću InsightFace (buffalo_l model)

    Kvantizacija i binarizacija embeddinga za generiranje diskretnih značajki

    Fuzzy Vault: zaključavanje (enrollment) i otključavanje (verifikacija) korištenjem polinoma i chaff točaka

    Automatsko preuzimanje modela buffalo_l pri prvom pokretanju

    Konfiguracija parametara putem config.ini datoteke

    Automatsko spremanje treniranog modela (preskače treniranje ako nema novih slika)

  INSTALACIJA
    
    pip install insightface==0.7.3 onnxruntime numpy opencv-python matplotlib scikit-learn sympy

  Prva upotreba i pokretanje

    Pripremite podatke:

        U data/training/ stavite slike različitih osoba za treniranje kvantizatora (10+ preporučeno)

        U data/gallery/ stavite po jednu sliku po korisniku za registraciju

        U data/probe/ stavite slike istih korisnika (ali različite slike) za test verifikacije

    Pokrenite sustav:

    python src/fuzzy_vault_face.py

    Model buffalo_l će se automatski preuzeti u weights/models/ pri prvom pokretanju (potreban internet).

    Prvo pokretanje potrajati će malo dulje zbog treniranja modela na slikama iz foldera /data/training, 
    to stanje se zatim pohrani u datoteku training_state.pkl koju program generira, te se pri sljedećim
    pokretanjima koda model ne trenira nego učitava postojeće stanje iz datoteke training_state.pkl.
    Ako se promijeni broj slika u folderu /data/training, sustav radi ponovno treniranje.

Konfiguracija

    Sve parametre (broj značajki, broj chaff točaka, stupanj polinoma, broj intervala...) možete mijenjati u config.ini datoteci

Kako radi sustav?

    Registracija: Iz slike korisnika generira se embedding, kvantizira i binarizira, pa se genuine točke zaključaju u fuzzy vault uz dodatak chaff točaka.

    Verifikacija: Nova slika korisnika prolazi isti postupak, a sustav pokušava otključati vault interpolacijom polinoma – uspjeh znači da je korisnik prepoznat.

    Sigurnost: Vault ne sadrži slike ni izravne biometrijske podatke; genuine i chaff točke su neprepoznatljive bez znanja ključa.

    Ispis: na kraju provjere, ispiše se broj točnih i lažnih potvrda, te broj točnih i netočnih odbijanja
