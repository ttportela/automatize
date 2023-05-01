### BiTULER: Bidirectional Trajectory-User Linking via Embedding and RNN Classifier

\[ *Runnable is included in Library* \]

You can run the classifier with the following command:
```bash
MAT-TC.py -ds "<DATASET_PREFIX>" -c "BITULER" --one-feature '<COLUMN_NAME>' "${DATAPATH}/${RUN}" "${DIR}"
```

Example:
```bash
MAT-TC.py -ds "specific" -c "BITULER"  --one-feature 'poi' "Datasets/Foursquare/run1" "BITULER-experiment01"
```


##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Identifying human mobility via trajectory embeddings | Gao, Q., Zhou, F., Zhang, K., Trajcevski, G., Luo, X., and Zhang, F. | 2017 | In IJCAI, volume 17 |  |  |
| Using Deep Learning for Trajectory Classification | Nicksson A. de Freitas.; Ticiana Coelho da Silva.; José Fernandes de Macêdo.; Leopoldo Melo Junior.; Matheus Cordeiro | 2021 | In Proceedings of the 13th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART | [Article](http://dx.doi.org/10.5220/0010227906640671) [Repository](https://github.com/nickssonfreitas/ICAART2021) |  |