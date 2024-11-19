# ToDo

- [ ] ベースライン

- [ ] どういうデータがあれば改善に繋がるか整理する
  - 各次元のエラー
    - ex. x_0 - pred-x_0, x_1 - pred-x_1, ...
    - [x] 実装
  - 画像にラベルの軌跡と予測の軌跡を重ねて可視化

- [x] 可視化してエラー分析できるようにする
  - [x] 読む <https://www.guruguru.science/competitions/25/discussions/b75b30bb-abcd-482d-b43a-fa325748e48d/>
  - [x] 実装

- [ ] 使える情報の活用
  - [x] 読む <https://www.guruguru.science/competitions/25/discussions/8b97734b-1f76-4075-b1af-5d227d6b70e8/>
  - [ ] 使えそうなアイデアを文章に起こす
  - [x] 新しい実験に組み込む
  - [ ] 結果の比較
  - [ ] 予測結果を眺めて言語化する

- [ ] NNベースモデル
  - [ ] 時系列画像と車両特徴量、信号機の情報を用いたベースラインモデル, 読む <https://www.guruguru.science/competitions/25/discussions/a85b8a5a-2041-4a1b-84fc-5ad5492c3978/>
    - [ ] 使えそうなアイデアを文章に起こす
    - [ ] 新しい実験に組み込む
    - [ ] 結果の比較
    - [ ] 予測結果を眺めて言語化する
  - [ ] LightGBM + CNN stacking baseline (CNN only), 読む <https://www.guruguru.science/competitions/25/discussions/03a365c7-27ce-490e-ab6f-e7788ce470c8/>
    - [ ] 使えそうなアイデアを文章に起こす
    - [ ] 新しい実験に組み込む
    - [ ] 結果の比較
    - [ ] 予測結果を眺めて言語化する
  - [ ] LightGBM + CNN stacking baseline (LightGBM + CNN), 読む <https://www.guruguru.science/competitions/25/discussions/30a373f3-3bde-4956-a636-1b8f0934750b/>
    - [ ] 使えそうなアイデアを文章に起こす
    - [ ] 新しい実験に組み込む
    - [ ] 結果の比較
    - [ ] 予測結果を眺めて言語化する

- [ ] Aux (exp007)
  - [x] vEgo
  - [x] aEgo
  - [x] brakePressed
  - [x] steeringAngleDeg
  - [x] steeringTorque
  - [ ] left&right blinker
  - [ ] has_trafficLight
  - [ ] trafficLightColor
  - [ ] trafficLightBbox
  - [ ] labelを画像上の軌道に変換してそれをラベルにする
  - [ ] flagはbinary classficationにする