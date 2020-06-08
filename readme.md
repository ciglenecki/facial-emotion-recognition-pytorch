# Prepoznavanje emocija iz izraza lica metodom deep learninga


![](https://www.researchgate.net/profile/Kevin_Bailly/publication/301830237/figure/fig4/AS:613889410605070@1523374049786/Examples-of-images-extracted-from-the-CK-dataset-DISFA-Dataset-The-Denver-Intensity-of.png)


## Uvod:

Strojnim učenjem stvorit ćemo istrenirani model koji može prepoznavati tj. klasificirati ljudske emocije uz određenu točnost, ovisno o pristupu kojeg koristimo prilikom odabira skupa podataka pomoću kojeg treniramo model.

Odabrani skup podataka Cohn-Kanade koji sadrži sekvence slika ljudskih emocija i njihove označene emocije.

Prije samog treniranja potrebno je procesirati i urediti podatke te stvoriti jedinku svake slike koja u sebi ima sve potrebne labele nužne za što bolje stvoren model.

To je uređen par (sadržaj slike, emocija, FACS podatak).

Za pojedinu sekvencu znamo samo o kojoj se emociji radi potrebno je linearno sklairati neutralnu emociju i emociju o kojoj se radi tako da svaka slika uistinu reprezentira o kojoj se kombinacija emocija radi.

## Treutna točnost modela
70% CK+ dataset

## Tools:

- PyTorch
- numpy
- pyplot
- PIL (Image module)

## Dataset:

- [Cohn-Kanade (CK+)](http://www.consortium.ri.cmu.edu/ckagree/)
  - 593 sequences from 123 subjects 
  - sequences go from neutral to peak emotion
  - 68 landmarks for each frame
    - Points are generated using Active Appearance Models (AAM)
  - emotions - 327/593 (55%) sequences are labeled with emotions
    - 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
    - Za AU detection - 123 različitih
    - Za emotion detection - 118 različitih
  - FACS
    - FACS je podskup bitnih landmarks AU obilježja koji se računa samo na peak frameu
    - Each line of the file corresponds to a specific AU and then the intensity. An example is given below
      - 0000000e+00   4.0000000e+00
      - 7000000e+01   2.0000000e+00
      - this means that AU9d and AU17b are present
      - B if an AU is present but the intensity is 0 this means that the intensity was not given

## Code summary (2020-03-11) :

- create\_image\_numpy.py
  - obrada strukture &quot;CK+&quot; dataseta
  - čitanje svih podataka u datasetu
  - stvaranje raw sample oblika: {image, emotion}
    - image: { ..pixels.. }
    - emotion: { [0.3, 0, 0, 0, 0, 0.7, 0] }
      - neutralna\_emocija + slika\_emocija = 1
  - sample se sprema kao &quot;subjekt\_sekvenca\_brojSekvence.npy&quot;
- create\_dataset.py
  - dohvat i čitanje svih .npy podataka
  - brada sample .npy podataka
  - mogućnost primjene TorchVision transformacije
    - image:
      - Face detection
        - facenet-pytorch [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
        - Pretrained face detection (MTCNN) and recognition (InceptionResnet) models
      - Random Horizontal Flip
      - Normalizacija
  - rekonstrukcija u PyTorch dataset format

## TODO-general:

- Image transformation:
  - Crop image to fixed 1:1 ratio
  - Augmentation:
    - Random rotation (45 max)
    - Lightning difference
- Dataset split (test/training/validation = 60/30/10):
  - Jednak omjer emocija mora biti zastupljen u sve 3 kategorije
    - sortirati slike
    - uzeti jednak udio emocija
- Training
  - one hot encoding
  - 5 cross validation
- Image classification
  - [AlexNet](https://arxiv.org/abs/1404.5997), [VGG](https://arxiv.org/abs/1409.1556), [ResNet](https://arxiv.org/abs/1512.03385)[SqueezeNet](https://arxiv.org/abs/1602.07360)[DenseNet](https://arxiv.org/abs/1608.06993)[Inception](https://arxiv.org/abs/1512.00567) v3 [GoogLeNet](https://arxiv.org/abs/1409.4842)[ShuffleNet](https://arxiv.org/abs/1807.11164) v2 [MobileNet](https://arxiv.org/abs/1801.04381) v2 [ResNeXt](https://arxiv.org/abs/1611.05431)[Wide ResNet](https://pytorch.org/docs/stable/torchvision/models.html#wide-resnet)[MNASNet](https://arxiv.org/abs/1807.11626)


## Single .npy file structure:
```
sample1{
    image: {..pixels..}
    emotions: {[0.3, 0, 0, 0, 0, 0.7, 0]}
}
```
## Dataset structure

```
------subject\_numberS005
------------sequence\_number001
------------------image\_numberS005\_001\_00000011.png


images{
    S005{
        001{
            S005\_001\_00000011: {..pixels..}
            S005\_001\_00000011: {..pixels..}
            ...
        }
        002{
            ...
        }
        ...
    }
    S993{
        ...
    }
}
```
