Semantic Artist Similarity Dataset

There are two datasets of artists with their biography text, and the list of top-10 most similar artists within the dataset: a corpus of 268 artists and a slightly larger dataset of 2,336 artists, both gathered from Last.fm. The former is mapped to the MIREX Audio and Music Similarity evaluation dataset, so that its similarity judgments can be used as ground truth. For the latter dataset we use the similarity between artists as provided by the Last.fm API. For every artist there is a list of top-10 most related artists. In the MIREX dataset there are only 188 artists with 10 similar artists, the other 80 have less than 10.

There are 4 files in the dataset. 

mirex_gold.txt and lastfmapi_gold.txt have the lists of similar artists for every artist of every dataset. Artists are identified by MusicBrainz ID. The format of the file is one line per artist, with the artist MBID separated by a tab with the list of top-10 related artists identified by their MBIDs separated by spaces.

artist_mbid \t artist_mbid_top10_list_separated_by_spaces \n

mb2uri_mirex and mb2uri_lastfm.txt have the list of artists. In each line there are three fields separated by tabs. First field is the MusicBrainz ID, second field is the last.fm name of the artist, and third field is the DBpedia uri.

artist_mbid \t lastfm_name \t dbpedia_uri \n

There are also 2 folders in the dataset with the biography texts of each dataset. Each .txt file in the biography folders is named with the MusicBrainz ID of the biographied artist. Biographies were gathered from the Last.fm wiki page of every artist.

Scientific References

For more details on how these files were generated, we refer to the following scientific publication. We would highly appreciate if scientific publications of works partly based on this dataset quote the following publication:

Oramas S., Sordo M., Espinosa-Anke L., Serra X. (2015). A Semantic-based approach for Artist Similarity. 16th International Society for Music Information Retrieval Conference ISMIR 2015


Compiler author of the data set

Conditions of Use

Dataset compiled and authored by Sergio Oramas and Mohamed Sordo. Copyright ©2015 Music Technology Group, Universitat Pompeu Fabra. All Rights Reserved.

The Artist Similarity dataset is offered free of charge for internal non-commercial use only. You may not redistribute, publically communicate or modify it. Please see the license terms in the README file within the dataset for applicable conditions.

Feedback

Problems, positive feedback, negative feedback... it is all welcome! Please help me improve the dataset by sending your feedback to:
sergio.oramas@upf.edu AND mtg-datasets@llista.upf.edu

In case of a problem report please include as many details as possible.
