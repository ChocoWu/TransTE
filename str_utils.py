

import re


def normalize_sent(sent):
    sent = str(sent).replace('``', '"')
    sent = str(sent).replace("''", '"')
    sent = str(sent).replace('-LRB-', '(')
    sent = str(sent).replace('-RRB-', ')')
    sent = str(sent).replace('-LSB-', '(')
    sent = str(sent).replace('-RSB-', ')')
    return sent


def collapse_role_type(role_type):
    '''
    collapse role types from 36 to 28 following Bishan Yang 2016
    we also have to handle types like 'Beneficiary#Recipient'
    :param role_type:
    :return:
    '''
    if role_type.startswith('Time-'):
        return 'Time'
    idx = role_type.find('#')
    if idx != -1:
        role_type = role_type[:idx]

    return role_type


# def normalize_tok(tok, lower=False, normalize_digits=False):
#
#     if lower:
#         tok = tok.lower()
#     if normalize_digits:
#         tok = re.sub(r"\d", "0", tok)
#         tok = re.sub(r"^(\d+[,])*(\d+)$", "0", tok)
#     return tok

def normalize_tok(tok, lower=True, normalize_digits=True):

    if lower:
        tok = normalize_sent(tok).lower().strip()
    if normalize_digits:
        RE_NUM = r"\b\d+(?:[\.,']\d+)?\b"
        RE_PERCENTAGE = RE_NUM + "%"
        tok = re.sub(r"\d", "0", tok)
        tok = re.sub(r"^(\d+[,])*(\d+)$", "0", tok)
        tok = re.sub(RE_NUM, '0', tok)
        tok = re.sub(RE_PERCENTAGE, '0', tok)
        tok = re.sub(r"%", 'percentage', tok)
        tok = re.sub(r"e\.g\.", 'for example', tok)
        tok = re.sub('0/0', '0', tok)

    return tok


def capitalize_first_char(sent):
    sent = str(sent[0]).upper() + sent[1:]
    return sent


if __name__ == '__main__':
    s = '0/0'
    print(normalize_tok(s))