# NLP
Enntity extraction model trainied using bert.
The model captures Parts of speech and entity tags for each word

Examples

## Sentences:

array([list(['Thousands', 'of', 'demonstrators', 'have', 'marched', 'through', 'London', 'to', 'protest', 'the', 'war', 'in', 'Iraq', 'and', 'demand', 'the', 'withdrawal', 'of', 'British', 'troops', 'from', 'that', 'country', '.']),
       list(['Iranian', 'officials', 'say', 'they', 'expect', 'to', 'get', 'access', 'to', 'sealed', 'sensitive', 'parts', 'of', 'the', 'plant', 'Wednesday', ',', 'after', 'an', 'IAEA', 'surveillance', 'system', 'begins', 'functioning', '.']),
       list(['Helicopter', 'gunships', 'Saturday', 'pounded', 'militant', 'hideouts', 'in', 'the', 'Orakzai', 'tribal', 'region', ',', 'where', 'many', 'Taliban', 'militants', 'are', 'believed', 'to', 'have', 'fled', 'to', 'avoid', 'an', 'earlier', 'military', 'offensive', 'in', 'nearby', 'South', 'Waziristan', '.'])],
      dtype=object)
      
## Parts of Speech:

array([list(['NNS', 'IN', 'NNS', 'VBP', 'VBN', 'IN', 'NNP', 'TO', 'VB', 'DT', 'NN', 'IN', 'NNP', 'CC', 'VB', 'DT', 'NN', 'IN', 'JJ', 'NNS', 'IN', 'DT', 'NN', '.']),
       list(['JJ', 'NNS', 'VBP', 'PRP', 'VBP', 'TO', 'VB', 'NN', 'TO', 'JJ', 'JJ', 'NNS', 'IN', 'DT', 'NN', 'NNP', ',', 'IN', 'DT', 'NNP', 'NN', 'NN', 'VBZ', 'VBG', '.']),
       list(['NN', 'NNS', 'NNP', 'VBD', 'JJ', 'NNS', 'IN', 'DT', 'NNP', 'JJ', 'NN', ',', 'WRB', 'JJ', 'NNP', 'NNS', 'VBP', 'VBN', 'TO', 'VB', 'VBN', 'TO', 'VB', 'DT', 'JJR', 'JJ', 'NN', 'IN', 'JJ', 'NNP', 'NNP', '.'])],
      dtype=object)
      
      
## Entity Tags

array([list(['O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'B-gpe', 'O', 'O', 'O', 'O', 'O']),
       list(['B-gpe', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-tim', 'O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O']),
       list(['O', 'O', 'B-tim', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O'])],
      dtype=object)
