import certifi
import json


class FMP_Data():
    def __init__(self):
        key = "94e168fc2df919c0706afd1c6fc1cac1"

    def get_data(self):
        urls = {
            "https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=94e168fc2df919c0706afd1c6fc1cac1"
        }
        data = []
        for u in urls:
            data += get_jsonparsed_data(u)
        return data


def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)
