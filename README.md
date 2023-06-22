MMS Dataset and Benchmark
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

Despite impressive advancements in multilingual corpora collection and
model training, developing large-scale deployments of multilingual
models still presents a significant challenge. This is particularly true
for language tasks that are culture-dependent. One such example is the
area of multilingual sentiment analysis, where affective markers can be
subtle and deeply ensconced in culture.

This work presents the most extensive open massively multilingual corpus
of datasets for training sentiment models. The corpus consists of 79
manually selected datasets from over 350 datasets reported in the
scientific literature based on strict quality criteria. The corpus
covers 27 languages representing 6 language families. Datasets can be
queried using several linguistic and functional features. In addition,
we present a multi-faceted sentiment classification benchmark
summarizing hundreds of experiments conducted on different base models,
training objectives, dataset collections, and fine-tuning strategies.

## Dataset

[Massively Multilingual Sentiment
Datasets](https://huggingface.co/datasets/Brand24/mms)

## Analysis and benchmarking

[HuggingFace Spaces with Analysis and
Benchmark](https://huggingface.co/spaces/Brand24/mms_benchmark)

## General statistics about the dataset

> It may take some time to download the dataset and generate train set
> inside HuggingFace dataset. Please be patient.

``` python
mms_dataset = datasets.load_dataset("Brand24/mms")
```

``` python
mms_dataset_df = mms_dataset["train"].to_pandas()
```

How many examples do we have?

``` python
mms_dataset.num_rows
```

    {'train': 6164762}

## Features

We provide not only texts and sentiment labels but we assigned many
additional dimensions for datasets and languages, hence it is possible
to splice and dice them as you want and need.

``` python
mms_dataset["train"].features
```

    {'_id': Value(dtype='int32', id=None),
     'text': Value(dtype='string', id=None),
     'label': ClassLabel(names=['negative', 'neutral', 'positive'], id=None),
     'original_dataset': Value(dtype='string', id=None),
     'domain': Value(dtype='string', id=None),
     'language': Value(dtype='string', id=None),
     'Family': Value(dtype='string', id=None),
     'Genus': Value(dtype='string', id=None),
     'Definite articles': Value(dtype='string', id=None),
     'Indefinite articles': Value(dtype='string', id=None),
     'Number of cases': Value(dtype='string', id=None),
     'Order of subject, object, verb': Value(dtype='string', id=None),
     'Negative morphemes': Value(dtype='string', id=None),
     'Polar questions': Value(dtype='string', id=None),
     'Position of negative word wrt SOV': Value(dtype='string', id=None),
     'Prefixing vs suffixing': Value(dtype='string', id=None),
     'Coding of nominal plurality': Value(dtype='string', id=None),
     'Grammatical genders': Value(dtype='string', id=None),
     'cleanlab_self_confidence': Value(dtype='float32', id=None)}

### Example

``` python
mms_dataset["train"][2001000]
```

    {'_id': 2001000,
     'text': 'I was a tomboy and this has such great memories for me. They fit exactly how I remember, PERFECTLY!!',
     'label': 2,
     'original_dataset': 'en_amazon',
     'domain': 'reviews',
     'language': 'en',
     'Family': 'Indo-European',
     'Genus': 'Germanic',
     'Definite articles': 'definite word distinct from demonstrative',
     'Indefinite articles': 'indefinite word distinct from one',
     'Number of cases': '2',
     'Order of subject, object, verb': 'SVO',
     'Negative morphemes': 'negative particle',
     'Polar questions': 'interrogative word order',
     'Position of negative word wrt SOV': 'SNegVO',
     'Prefixing vs suffixing': 'strongly suffixing',
     'Coding of nominal plurality': 'plural suffix',
     'Grammatical genders': 'no grammatical gender',
     'cleanlab_self_confidence': 0.9978116750717163}

### Classes

``` python
labels = mms_dataset["train"].features["label"].names
labels
```

    ['negative', 'neutral', 'positive']

``` python
mms_dataset_df["label_name"] = mms_dataset_df["label"].apply(lambda x: labels[x])
```

### Classes distribution

``` python
labels_stats_df = pd.DataFrame(mms_dataset_df.label_name.value_counts())
labels_stats_df["percentage"] = (labels_stats_df["label_name"] / labels_stats_df["label_name"].sum()).round(3)
labels_stats_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label_name</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>positive</th>
      <td>3494478</td>
      <td>0.567</td>
    </tr>
    <tr>
      <th>neutral</th>
      <td>1341354</td>
      <td>0.218</td>
    </tr>
    <tr>
      <th>negative</th>
      <td>1328930</td>
      <td>0.216</td>
    </tr>
  </tbody>
</table>
</div>

## Sentiment orientation for each language

``` python
cols = ['language', 'label_name']
mms_dataset_df[cols].value_counts().to_frame().reset_index().rename(columns={0: 'count'}).sort_values(by=cols, ascending=True)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language</th>
      <th>label_name</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>ar</td>
      <td>negative</td>
      <td>138899</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ar</td>
      <td>neutral</td>
      <td>192774</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ar</td>
      <td>positive</td>
      <td>600402</td>
    </tr>
    <tr>
      <th>53</th>
      <td>bg</td>
      <td>negative</td>
      <td>13930</td>
    </tr>
    <tr>
      <th>41</th>
      <td>bg</td>
      <td>neutral</td>
      <td>28657</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62</th>
      <td>ur</td>
      <td>neutral</td>
      <td>8585</td>
    </tr>
    <tr>
      <th>67</th>
      <td>ur</td>
      <td>positive</td>
      <td>5836</td>
    </tr>
    <tr>
      <th>9</th>
      <td>zh</td>
      <td>negative</td>
      <td>117967</td>
    </tr>
    <tr>
      <th>21</th>
      <td>zh</td>
      <td>neutral</td>
      <td>69016</td>
    </tr>
    <tr>
      <th>6</th>
      <td>zh</td>
      <td>positive</td>
      <td>144719</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 3 columns</p>
</div>

## Per language

``` python
cols = ['language']
mms_dataset_df[cols].value_counts().to_frame().reset_index().rename(columns={0: 'count'}).sort_values(by=cols, ascending=True)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ar</td>
      <td>932075</td>
    </tr>
    <tr>
      <th>15</th>
      <td>bg</td>
      <td>62150</td>
    </tr>
    <tr>
      <th>20</th>
      <td>bs</td>
      <td>36183</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cs</td>
      <td>196287</td>
    </tr>
    <tr>
      <th>4</th>
      <td>de</td>
      <td>315887</td>
    </tr>
    <tr>
      <th>0</th>
      <td>en</td>
      <td>2330486</td>
    </tr>
    <tr>
      <th>2</th>
      <td>es</td>
      <td>418712</td>
    </tr>
    <tr>
      <th>23</th>
      <td>fa</td>
      <td>13525</td>
    </tr>
    <tr>
      <th>6</th>
      <td>fr</td>
      <td>210631</td>
    </tr>
    <tr>
      <th>25</th>
      <td>he</td>
      <td>8619</td>
    </tr>
    <tr>
      <th>22</th>
      <td>hi</td>
      <td>16999</td>
    </tr>
    <tr>
      <th>12</th>
      <td>hr</td>
      <td>77594</td>
    </tr>
    <tr>
      <th>16</th>
      <td>hu</td>
      <td>56682</td>
    </tr>
    <tr>
      <th>24</th>
      <td>it</td>
      <td>12065</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ja</td>
      <td>209780</td>
    </tr>
    <tr>
      <th>26</th>
      <td>lv</td>
      <td>5790</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pl</td>
      <td>236688</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pt</td>
      <td>157834</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ru</td>
      <td>110930</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sk</td>
      <td>56623</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sl</td>
      <td>113543</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sq</td>
      <td>44284</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sr</td>
      <td>76368</td>
    </tr>
    <tr>
      <th>19</th>
      <td>sv</td>
      <td>41346</td>
    </tr>
    <tr>
      <th>14</th>
      <td>th</td>
      <td>72319</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ur</td>
      <td>19660</td>
    </tr>
    <tr>
      <th>3</th>
      <td>zh</td>
      <td>331702</td>
    </tr>
  </tbody>
</table>
</div>

## Example of filtering datasets

### Choose only Polish

``` python
pl = mms_dataset.filter(lambda row: row['language'] == 'pl')
```

    Filter:   0%|          | 0/6164762 [00:00<?, ? examples/s]

``` python
pl["train"].to_pandas().sample(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_id</th>
      <th>text</th>
      <th>label</th>
      <th>original_dataset</th>
      <th>domain</th>
      <th>language</th>
      <th>Family</th>
      <th>Genus</th>
      <th>Definite articles</th>
      <th>Indefinite articles</th>
      <th>Number of cases</th>
      <th>Order of subject, object, verb</th>
      <th>Negative morphemes</th>
      <th>Polar questions</th>
      <th>Position of negative word wrt SOV</th>
      <th>Prefixing vs suffixing</th>
      <th>Coding of nominal plurality</th>
      <th>Grammatical genders</th>
      <th>cleanlab_self_confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>215921</th>
      <td>5119386</td>
      <td>Typujcie jaki dziś będzie wynik St.Pats - Legi...</td>
      <td>2</td>
      <td>pl_twitter_sentiment</td>
      <td>social_media</td>
      <td>pl</td>
      <td>Indo-European</td>
      <td>Slavic</td>
      <td>no article</td>
      <td>no article</td>
      <td>6-7</td>
      <td>SVO</td>
      <td>negative particle</td>
      <td>question particle</td>
      <td>SNegVO</td>
      <td>strongly suffixing</td>
      <td>plural suffix</td>
      <td>masculine, feminine, neuter</td>
      <td>0.589098</td>
    </tr>
    <tr>
      <th>86525</th>
      <td>4989990</td>
      <td>@KaczmarSF Przyjemne ciarki mam, gdy patrzę na...</td>
      <td>2</td>
      <td>pl_twitter_sentiment</td>
      <td>social_media</td>
      <td>pl</td>
      <td>Indo-European</td>
      <td>Slavic</td>
      <td>no article</td>
      <td>no article</td>
      <td>6-7</td>
      <td>SVO</td>
      <td>negative particle</td>
      <td>question particle</td>
      <td>SNegVO</td>
      <td>strongly suffixing</td>
      <td>plural suffix</td>
      <td>masculine, feminine, neuter</td>
      <td>0.950756</td>
    </tr>
    <tr>
      <th>66031</th>
      <td>4969496</td>
      <td>szkoda bylo czasu i kasy .</td>
      <td>0</td>
      <td>pl_polemo</td>
      <td>reviews</td>
      <td>pl</td>
      <td>Indo-European</td>
      <td>Slavic</td>
      <td>no article</td>
      <td>no article</td>
      <td>6-7</td>
      <td>SVO</td>
      <td>negative particle</td>
      <td>question particle</td>
      <td>SNegVO</td>
      <td>strongly suffixing</td>
      <td>plural suffix</td>
      <td>masculine, feminine, neuter</td>
      <td>0.940540</td>
    </tr>
    <tr>
      <th>137768</th>
      <td>5041233</td>
      <td>@shinyvalentine mam ja w dupie lecz bylo to kr...</td>
      <td>0</td>
      <td>pl_twitter_sentiment</td>
      <td>social_media</td>
      <td>pl</td>
      <td>Indo-European</td>
      <td>Slavic</td>
      <td>no article</td>
      <td>no article</td>
      <td>6-7</td>
      <td>SVO</td>
      <td>negative particle</td>
      <td>question particle</td>
      <td>SNegVO</td>
      <td>strongly suffixing</td>
      <td>plural suffix</td>
      <td>masculine, feminine, neuter</td>
      <td>0.220028</td>
    </tr>
    <tr>
      <th>118766</th>
      <td>5022231</td>
      <td>@itiNieWracaj pokazują to gdzieś?</td>
      <td>2</td>
      <td>pl_twitter_sentiment</td>
      <td>social_media</td>
      <td>pl</td>
      <td>Indo-European</td>
      <td>Slavic</td>
      <td>no article</td>
      <td>no article</td>
      <td>6-7</td>
      <td>SVO</td>
      <td>negative particle</td>
      <td>question particle</td>
      <td>SNegVO</td>
      <td>strongly suffixing</td>
      <td>plural suffix</td>
      <td>masculine, feminine, neuter</td>
      <td>0.139179</td>
    </tr>
  </tbody>
</table>
</div>

## Use cases

### Case 1

Thus, when training a sentiment classifier using our dataset, one may
download different facets of the collection. For instance, one can
download all datasets in `Slavic` languages in which polar questions are
formed using the interrogative word order or download all datasets from
the `Afro-Asiatic` language family with no morphological case-making.

``` python
slavic = mms_dataset.filter(lambda row: row["Genus"] == "Slavic" and row["Polar questions"] == "interrogative word order")
```

    Filter:   0%|          | 0/6164762 [00:00<?, ? examples/s]

``` python
slavic
```

    DatasetDict({
        train: Dataset({
            features: ['_id', 'text', 'label', 'original_dataset', 'domain', 'language', 'Family', 'Genus', 'Definite articles', 'Indefinite articles', 'Number of cases', 'Order of subject, object, verb', 'Negative morphemes', 'Polar questions', 'Position of negative word wrt SOV', 'Prefixing vs suffixing', 'Coding of nominal plurality', 'Grammatical genders', 'cleanlab_self_confidence'],
            num_rows: 252910
        })
    })

### Case 2

``` python
afro_asiatic = mms_dataset.filter(lambda row: row["Family"] == "Afro-Asiatic" and row["Number of cases"] == "no morphological case-making")
```

    Filter:   0%|          | 0/6164762 [00:00<?, ? examples/s]

``` python
afro_asiatic
```

    DatasetDict({
        train: Dataset({
            features: ['_id', 'text', 'label', 'original_dataset', 'domain', 'language', 'Family', 'Genus', 'Definite articles', 'Indefinite articles', 'Number of cases', 'Order of subject, object, verb', 'Negative morphemes', 'Polar questions', 'Position of negative word wrt SOV', 'Prefixing vs suffixing', 'Coding of nominal plurality', 'Grammatical genders', 'cleanlab_self_confidence'],
            num_rows: 8619
        })
    })

## Dataset Curators

The corpus was put together by

- [@laugustyniak](https://www.linkedin.com/in/lukaszaugustyniak/)
- [@swozniak](https://www.linkedin.com/in/wscode/)
- [@mgruza](https://www.linkedin.com/in/marcin-gruza-276b2512b/)
- [@pgramacki](https://www.linkedin.com/in/piotrgramacki/)
- [@krajda](https://www.linkedin.com/in/krzysztof-rajda/)
- [@mmorzy](https://www.linkedin.com/in/mikolajmorzy/)
- [@tkajdanowicz](https://www.linkedin.com/in/kajdanowicz/)

## Citation

``` bibtex
@misc{augustyniak2023massively,
      title={Massively Multilingual Corpus of Sentiment Datasets and Multi-faceted Sentiment Classification Benchmark}, 
      author={Łukasz Augustyniak and Szymon Woźniak and Marcin Gruza and Piotr Gramacki and Krzysztof Rajda and Mikołaj Morzy and Tomasz Kajdanowicz},
      year={2023},
      eprint={2306.07902},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements

- BRAND24 - https://brand24.com
- CLARIN-PL-Biz - https://clarin.biz

## Licensing Information

These data are released under this licensing scheme. We do not own any
text from which these data and datasets have been extracted.

We license the actual packaging of these data under the
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/4.0/

This work is published from Poland.

Should you consider that our data contains material that is owned by you
and should, therefore not be reproduced here, please: \* Clearly
identify yourself with detailed contact data such as an address,
telephone number, or email address at which you can be contacted. \*
Clearly identify the copyrighted work claimed to be infringed. \*
Clearly identify the material claimed to be infringing and the
information reasonably sufficient to allow us to locate the material.

We will comply with legitimate requests by removing the affected sources
from the next release of the corpus.
