## 2024/11/17

- ベースライン書いたので学習を回し始めた
- ちょっと遅いのでどうにかしたい
- 可視化した感じ、ベースラインだとカーブが難しそう
- なめらかさがないような予測も見られるからsmoothingもできたら良いかもしれない

## 2024/11/18

- convnext強い、resnet34d,tf_efficientnet_b3_nsよりも二回りくらい強い

## 2024/11/19

- exp007

Aux: vEgo(0.5),aEgo(0.5),rightBlinker(0.1),leftBlinker(0.1),brake(0.1)
Scores: [0.739008, 0.752076, 0.789385, 0.786481, 0.676495] Mean: 0.748689

Aux: vEgo(0.5),aEgo(0.5),rightBlinker(0.5),leftBlinker(0.5),brake(0.5)
Scores: [0.751432, 0.780632, 0.758083, 0.761099, 0.737548] Mean: 0.7577588

Aux: vEgo(0.5),aEgo(0.5),rightBlinker(0.1),leftBlinker(0.1),brake(0.1),steeringAgreeDeg(0.01)
Scores: [0.753489, 0.769176, 0.732074, 0.76733, 0.698637] Mean: 0.7441412
Scores: [0.7465812563896179, 0.7679423093795776, 0.7398396730422974, 0.7303324937820435, 0.6915957927703857], Mean: 0.7352583050727844 +/- 0.025092313141482372
Scores: [0.7465812563896179, 0.7679423093795776, 0.7398396730422974, 0.7303324937820435, 0.6915957927703857], Mean: 0.7352583050727844 +/- 0.025092313141482372
ScoresFold: 0.735254168510437

AUXの微妙な違いで、ensembleのタネになりそう

augmentationを足しても意味なかったっぽい。。 < なんと渡してなかったwwwww

Aux: vEgo(0.5),aEgo(0.5),rightBlinker(0.1),leftBlinker(0.1),brake(0.1),steeringAgreeDeg(0.01),steeringTorque(0.01)
Scores: [0.7314547896385193, 0.7737607359886169, 0.7413656711578369, 0.7899930477142334, 0.7404636740684509], Mean: 0.7554075837135314 +/- 0.022482102242799125

- Augmentationを追加した

```python
train_tranforms: list = [
    albu.Resize(size, size),
    albu.OneOf([
        albu.GaussNoise(var_limit=(10, 50)),
        albu.GaussianBlur(),
        albu.MotionBlur(),
    ]),
    albu.OneOf([
        albu.RandomGamma(gamma_limit=(30, 150), p=1),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=1),
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        albu.CLAHE(clip_limit=5.0, tile_grid_size=(5, 5), p=1),
    ]),
    albu.HorizontalFlip(p=0.5),
    albu.ShiftScaleRotate(
        shift_limit=0.0,
        scale_limit=0.1,
        rotate_limit=15,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.8,
    ),
    albu.CoarseDropout(max_height=50, max_width=50, min_holes=2, p=0.5),
    albu.Normalize(mean=[0] * 9, std=[1] * 9),
    ToTensorV2(),
]
```

Scores: [0.5574209094047546, 0.5814367532730103, 0.5712170600891113, 0.5709481835365295, 0.5701022148132324], Mean: 0.5702250242233277 +/- 0.007631125752773015
ScoresFold: 0.5702422261238098

```python
train_tranforms: list = [
    albu.Resize(size, size),
    albu.OneOf([
        albu.GaussNoise(var_limit=(10, 50)),
        albu.GaussianBlur(),ｊ
        albu.MotionBlur(),
    ]),
    albu.Normalize(mean=[0] * 9, std=[1] * 9),
    ToTensorV2(),
]
```
Scores: [0.5580177307128906, 0.5688067078590393, 0.5675504803657532, 0.5719417929649353, 0.5585517883300781], Mean: 0.5649737000465394 +/- 0.005648230237663063
ScoresFold: 0.5649837255477905

```python
train_tranforms: list = [
    albu.Resize(size, size),
    albu.OneOf([
        albu.GaussNoise(var_limit=(10, 50)),
        albu.GaussianBlur(),
        albu.MotionBlur(),
    ]),
    albu.OneOf([
        albu.RandomGamma(gamma_limit=(30, 150), p=1),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=1),
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        albu.CLAHE(clip_limit=5.0, tile_grid_size=(5, 5), p=1),
    ]),
    albu.CoarseDropout(max_height=50, max_width=50, min_holes=2, p=0.5),
    albu.Normalize(mean=[0] * 9, std=[1] * 9),
    ToTensorV2(),
]
```

Scores: [0.5608667731285095, 0.5692499279975891, 0.57230544090271, 0.5484209656715393, 0.5538298487663269], Mean: 0.560934591293335 +/- 0.009005707492753222
ScoresFold: 0.5609493851661682

## 2024/11/21

### exp007 train_gbdt

base:
    Total Score: total_score = 0.22430463135242462

label一つずつ
Total Score: total_score = 0.23619685967125298
Scores: [0.23282591591180538, 0.23596476934262917, 0.24003224431898812, 0.23727593380463946, 0.2348471729812392], Mean: 0.23618920727186027 +/- 0.002413009137841947

==========================
LightGBM for each cols.

{'x_0': 0.06338603460008993,
 'x_1': 0.1355204368980031,
 'x_2': 0.23106604265515954,
 'x_3': 0.35820086822980785,
 'x_4': 0.5129338811140567,
 'x_5': 0.6870242650924265,
 'y_0': 0.03131077851282361,
 'y_1': 0.07086407736000334,
 'y_2': 0.12326019662733646,
 'y_3': 0.19762026280575296,
 'y_4': 0.2940380787841482,
 'y_5': 0.41983997857345595,
 'z_0': 0.024973007554025053,
 'z_1': 0.05132053040368659,
 'z_2': 0.07835185184404217,
 'z_3': 0.10618095770453051,
 'z_4': 0.1348927238975363,
 'z_5': 0.16592361232760644}

Total Score: total_score = 0.2049556800389284
Scores: [0.20237828863479027, 0.20347440311863732, 0.20544336590748172, 0.20865146606813254, 0.20481708805469392]
Mean: 0.20495292235674717 +/- 0.002133212101915711

Training finished. CALLED_TIME = '20241122-10:55:35', COMMIT_HASH = 'f65b1acf6163d53192cec8564e1dc246bd397453'
==========================

### exp007 train

今の実装とDisscussionとの検証をする。
とりあえず今の実装の結果のログをとる。

===================================================
Exp: exp007, DESC:
simple baseline

Total Score: 0.5553814172744751

Scores: [0.5564802885055542, 0.5657303333282471, 0.5564661622047424, 0.5585641264915466, 0.5396636128425598]
Mean: 0.55538090467453 +/- 0.008563448672594792

Training finished. CALLED_TIME = '20241122-12:10:32', COMMIT_HASH = 'f65b1acf6163d53192cec8564e1dc246bd397453', Duration: 131 [min], 7918 [sec]
===================================================

trafic_lightに同じようなラベルがある
多分、t,t-0.5,とかのラベル全部同じ画像でついてる. これはindexで順番がついてる

```
green [46  6 47  7]
[159, 197, 64]
green [46  6 48  7]
[159, 197, 64]
green [46  6 47  7]
[159, 197, 64]
green [47  6 48  7]
[159, 197, 64]
```

- encoderをresnet34dにしてconvnext_tinyとの差を測る

- modelどっちが大きいんだっけ？

resnet34d
===============================================================================================
Total params: 11,208,153
Trainable params: 11,208,153
Non-trainable params: 0
Total mult-adds (G): 2.68
===============================================================================================
Input size (MB): 2.36
Forward/backward pass size (MB): 51.91
Params size (MB): 44.83
Estimated Total Size (MB): 99.10
===============================================================================================

convnext
=========================================================================================================
Total params: 27,848,569
Trainable params: 27,848,569
Non-trainable params: 0
Total mult-adds (M): 631.51
=========================================================================================================
Input size (MB): 2.36
Forward/backward pass size (MB): 171.49
Params size (MB): 111.37
Estimated Total Size (MB): 285.22
=========================================================================================================

convnextのが圧倒的によさげ