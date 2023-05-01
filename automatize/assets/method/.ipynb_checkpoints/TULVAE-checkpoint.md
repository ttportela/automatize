### TULVAE: Trajectory-User Linking via Variational Autoencoder Classifier

\[ *Runnable is included in Library* \]

You can run the classifier with the following command:
```bash
MAT-TC.py -ds "<DATASET_PREFIX>" -c "TULVAE" --one-feature '<COLUMN_NAME>' "${DATAPATH}/${RUN}" "${DIR}"
```

Example:
```bash
MAT-TC.py -ds "specific" -c "TULVAE" --one-feature 'poi' "Datasets/Foursquare/run1" "TULVAE-experiment01"
```


##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Trajectory-user linking via variational autoencoder | Zhou, Fan; Gao, Qiang; Trajcevski, Goce; Zhang, Kunpeng; Zhong, Ting; Zhang, Fengli. | 2018 | IJCAI International Joint Conference on Artificial Intelligence |  |  |
| Using Deep Learning for Trajectory Classification | Nicksson A. de Freitas.; Ticiana Coelho da Silva.; José Fernandes de Macêdo.; Leopoldo Melo Junior.; Matheus Cordeiro | 2021 | In Proceedings of the 13th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART | [Article](http://dx.doi.org/10.5220/0010227906640671) [Repository](https://github.com/nickssonfreitas/ICAART2021) |  |