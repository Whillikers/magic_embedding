# code_mappings

Various mappings from cards to integers and back again.
All mappings are stored as pickled dicts.
"Encodings" are maps from card names to integers, stored as {'card_name': code}.
"decoding" means a map from integers to card names, stored as {code: 'card_name'}

All mappings are sorted by decreasing card popularity, and the suffix indicates how many cards are considered.
Restricting the model to only the most popular cards is useful to eliminate recommending cards that aren't in enough decks to be represented well.
