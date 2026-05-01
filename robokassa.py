import hashlib
from urllib.parse import urlencode
from config import (
    ROBOKASSA_LOGIN, ROBOKASSA_PASSWORD1, ROBOKASSA_PASSWORD2,
    ROBOKASSA_TEST_MODE, ROBOKASSA_RESULT_URL, ROBOKASSA_SUCCESS_URL,
    ROBOKASSA_FAIL_URL
)

def get_payment_url(inv_id: int, amount: float, description: str = "Пополнение баланса") -> str:
    # ВСЕГДА используем боевой домен, тестовый режим управляется параметром IsTest
    base_url = "https://merchant.robokassa.ru/Index.aspx"

    params = {
        "MerchantLogin": ROBOKASSA_LOGIN,
        "InvId": inv_id,
        "OutSum": f"{amount:.2f}",
        "Description": description,
        "SignatureValue": _make_signature(inv_id, amount, ROBOKASSA_PASSWORD1),
        "ResultURL": ROBOKASSA_RESULT_URL,
        "SuccessURL": ROBOKASSA_SUCCESS_URL,
        "FailURL": ROBOKASSA_FAIL_URL,
        "IsTest": 1 if ROBOKASSA_TEST_MODE else 0,
        "Encoding": "utf-8",
    }
    params = {k: v for k, v in params.items() if v is not None}
    query = urlencode(params)
    return f"{base_url}?{query}"

def _make_signature(inv_id: int, amount: float, password: str) -> str:
    sign_str = f"{ROBOKASSA_LOGIN}:{amount:.2f}:{inv_id}:{password}"
    return hashlib.md5(sign_str.encode('utf-8')).hexdigest().upper()

def check_result_signature(params: dict) -> bool:
    out_sum = params.get("OutSum")
    inv_id = params.get("InvId")
    signature = params.get("SignatureValue")
    if not all([out_sum, inv_id, signature]):
        return False
    sign_str = f"{out_sum}:{inv_id}:{ROBOKASSA_PASSWORD2}"
    expected = hashlib.md5(sign_str.encode('utf-8')).hexdigest().upper()
    return signature.upper() == expected

def check_success_signature(params: dict) -> bool:
    out_sum = params.get("OutSum")
    inv_id = params.get("InvId")
    signature = params.get("SignatureValue")
    if not all([out_sum, inv_id, signature]):
        return False
    sign_str = f"{out_sum}:{inv_id}:{ROBOKASSA_PASSWORD1}"
    expected = hashlib.md5(sign_str.encode('utf-8')).hexdigest().upper()
    return signature.upper() == expected
