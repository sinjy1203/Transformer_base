SPECIAL_TOKENS = {"UNK": "<unk>", "PAD": "<pad>", "SOS": "<sos>", "EOS": "<eos>"}
SPECIAL_TOKENS_ORDER = ["UNK", "PAD", "SOS", "EOS"]
SPECIAL_TOKENS_IDX = {token: i for i, token in enumerate(SPECIAL_TOKENS_ORDER)}
