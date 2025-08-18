# 2025-08-11 08:00:00 외부 호출용 Facade 클래스 추가
# 2025-08-11 20:50:09 Java를 Python으로 변환함

"""
/*
* GUID를 생성
*
*/
"""

import time
import threading
import random
import socket
import base64
import hmac
import hashlib
from typing import Optional


class GeneRateUtils:
    # Java: private static volatile String hexServerIP = null;
    hexServerIP = None

    # Java: private static final ThreadLocal threadLocalSeeder = ThreadLocal.withInitial(SecureRandom::new);
    threadLocalSeeder = threading.local()

    # Java의 synchronized 대체 (동일 로직 유지 목적)
    _ip_lock = threading.Lock()

    @staticmethod
    def generateGuid(o):
        if GeneRateUtils.hexServerIP is None:
            # Java: synchronized (GeneRateUtils.class) { ... }
            with GeneRateUtils._ip_lock:
                if GeneRateUtils.hexServerIP is None:
                    GeneRateUtils.hexServerIP = GeneRateUtils.resolveHexServerIP()

        # Java: (int) System.currentTimeMillis();
        timeLow = int(time.time() * 1000) & 0xFFFFFFFF

        # Java: System.identityHashCode(o)
        if o is None:
            ident = 0
        else:
            ident = id(o) & 0xFFFFFFFF
        objectHash = GeneRateUtils.hexFormat(ident, 8)

        # Java: threadLocalSeeder.get().nextInt();
        rng = getattr(GeneRateUtils.threadLocalSeeder, "rand", None)
        if rng is None:
            rng = random.SystemRandom()
            GeneRateUtils.threadLocalSeeder.rand = rng
        node = rng.getrandbits(32)

        parts = [
            GeneRateUtils.hexFormat(timeLow, 8),
            GeneRateUtils.hexServerIP,
            objectHash,
            GeneRateUtils.hexFormat(node, 8),
        ]
        return "".join(parts)

    @staticmethod
    def resolveHexServerIP():
        try:
            # Java: InetAddress.getByName("127.0.0.1").getAddress();
            ip_bytes = socket.inet_aton("127.0.0.1")  # network byte order (big-endian) 4 bytes
            return GeneRateUtils.hexFormat(GeneRateUtils.bytesToInt(ip_bytes), 8)
        except Exception as e:
            # Java: throw new RuntimeException("서버 IP 확인 실패", e);
            raise RuntimeError("서버 IP 확인 실패") from e

    @staticmethod
    def bytesToInt(b):
        result = 0
        for byte in b:
            result = (result << 8) | (byte & 0xFF)
        return result

    @staticmethod
    def hexFormat(value, length):
        # Java: Integer.toHexString(value)와 동일하게 32비트로 마스킹 후 소문자 16진수
        hex_str = format(value & 0xFFFFFFFF, "x")
        if len(hex_str) < length:
            hex_str = ("0" * (length - len(hex_str))) + hex_str
        return hex_str

    @staticmethod
    def get16ByteKeyBySha256(key, data):
        """
        HMAC-SHA256 기반 16바이트 키 생성
        """
        try:
            # Java: Mac.getInstance("HmacSHA256") + SecretKeySpec
            mac = hmac.new(key.encode("utf-8"), msg=data.encode("utf-8"), digestmod=hashlib.sha256)
            hash_bytes = mac.digest()
            # Java: Base64.getEncoder().encodeToString(hashBytes);
            return base64.b64encode(hash_bytes).decode("ascii")
        except Exception as e:
            # Java: throw new RuntimeException("SHA256 해시 생성 실패", e);
            raise RuntimeError("SHA256 해시 생성 실패") from e


class GeneRateUtilsFacade:
    """
    외부 모듈에서 인스턴스 형태로 호출할 수 있도록 제공하는 래퍼 클래스입니다.
    기존 GeneRateUtils의 정적 메서드를 그대로 위임하며, 선택적으로 서버 IP를 지정할 수 있습니다.
    원본 주석과 구현은 변경하지 않습니다.
    """

    def __init__(self, server_ip: Optional[str] = None):
        # server_ip가 주어지면 GeneRateUtils.hexServerIP를 해당 값으로 고정합니다.
        if server_ip:
            try:
                ip_bytes = socket.inet_aton(server_ip)
                GeneRateUtils.hexServerIP = GeneRateUtils.hexFormat(
                    GeneRateUtils.bytesToInt(ip_bytes), 8
                )
            except Exception as e:
                raise RuntimeError("서버 IP 설정 실패") from e

    def generateGuid(self, o):
        return GeneRateUtils.generateGuid(o)

    def get16ByteKeyBySha256(self, key: str, data: str) -> str:
        return GeneRateUtils.get16ByteKeyBySha256(key, data)
    
__all__ = ['GeneRateUtils', 'GeneRateUtilsFacade']