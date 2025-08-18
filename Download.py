import httpx
import atexit
import asyncio
from pathlib import Path,PurePath
from typing import Optional
from loguru import logger


class Download:
    """이미지 파일 다운로드 및 저장을 담당하는 유틸리티 클래스"""

    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    def _get_client(cls) -> httpx.AsyncClient:
        if cls._client is None:
            # limits 설정으로 커넥션/메모리 상한도 걸어둠
            cls._client = httpx.AsyncClient(timeout=30.0,
                                            limits=httpx.Limits(max_keepalive_connections=50,
                                                                max_connections=100))
        return cls._client
    
    @classmethod
    async def close(cls) -> None:
        """
        애플리케이션 종료 시 연결 풀을 안전하게 닫는다.
        """
        if cls._client is not None:
            await cls._client.aclose()
            cls._client = None

    @staticmethod
    async def save(
        *,
        file_url: str,
        download_dir: Path,
        file_name: str,
        tran_id: str
    ) -> Optional[Path]:
        """비동기적으로 이미지를 다운받아 지정된 디렉터리에 저장한다.

        Args:
            file_url (str): 다운로드할 파일의 URL.
            download_dir (Path): 파일을 저장할 디렉터리.
            file_name (str): 저장 시 사용할 파일명.
            tran_id (str): 트랜잭션 식별자(로깅용).

        Returns:
            Optional[Path]: 저장된 파일의 경로(성공 시) 또는 None(실패 시).
        """
        # 저장 경로 준비
        download_dir.mkdir(parents=True, exist_ok=True)
        clean_name = PurePath(file_name).name      # 디렉터리 구간 제거
        if clean_name != file_name or "/" in clean_name or "\\" in clean_name:
            raise ValueError("허용되지 않는 파일 이름")
        save_path = download_dir / clean_name
        client = Download._get_client()

        logger.info(f"[배경작업][{tran_id}] 이미지 다운로드 시도: {file_url} -> {save_path}")
        try:
            resp = await client.get(file_url)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            logger.info(f"[배경작업][{tran_id}] 이미지 다운로드 및 저장 성공: {save_path}")
            return save_path
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[배경작업][{tran_id}] 이미지 다운로드 HTTP 오류: {e.request.url} - {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(
                f"[배경작업][{tran_id}] 이미지 다운로드 요청 오류: {e.request.url} - {e}"
            )
        except IOError as e:
            logger.error(
                f"[배경작업][{tran_id}] 이미지 파일 저장 실패: {save_path}, 오류: {e}"
            )
        except Exception as e:
            logger.error(
                f"[배경작업][{tran_id}] 이미지 다운로드/저장 중 알 수 없는 오류: {file_url}, 오류: {e}",
                exc_info=True,
            )
        return None 
    # 프로세스 종료 시 비동기 클라이언트를 닫는다.

    @staticmethod
    def checkMMS(saved_file: str, mms_nas_dir: str) -> bool:
        """
        saved_file 이 MMS_NAS_DIR 내에 실제 존재하는지 확인
        """
        try:

            if not mms_nas_dir:
                logger.error("MMS_NAS_DIR 설정이 없습니다.")
                return False
            file_path = Path(mms_nas_dir) / saved_file
            return file_path.exists()
        except Exception as e:
            logger.error(f"MMS 파일 체크 중 오류 발생: {e}")
            return False
atexit.register(lambda: asyncio.run(Download.close()))