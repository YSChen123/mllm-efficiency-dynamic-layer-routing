"""Bounded Hub backoff and user-authorized selection by available GPU memory."""
import datetime
from email.utils import parsedate_to_datetime
import re
import time


def retry_hub(operation, attempts=3, sleep=time.sleep, emit=print):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, 'response', None)
            if response is None or response.status_code != 429 or attempt + 1 == attempts:
                raise
            headers = {key.lower(): value for key, value in response.headers.items()}
            delays = [300.0]
            retry_after = headers.get('retry-after', '')
            if retry_after:
                try:
                    delays.append(float(retry_after))
                except ValueError:
                    delays.append((parsedate_to_datetime(retry_after) - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
            reset = re.search(r'(?:^|;)\s*t=(\d+)', headers.get('ratelimit', ''))
            if reset:
                delays.append(float(reset.group(1)))
            delay = max(delays) + 5
            emit(f'WAITING_HUB_RATE_LIMIT seconds={delay} retry={attempt + 1}/{attempts - 1}', flush=True) if emit is print else emit(f'WAITING_HUB_RATE_LIMIT seconds={delay}')
            sleep(delay)


def select_gpu(csv_rows, minimum_mib=26624):
    rows = []
    for line in csv_rows.strip().splitlines():
        index, total, free, utilization = (value.strip() for value in line.split(','))
        rows.append(dict(index=index, total_mib=int(total), free_mib=int(free), utilization=int(utilization)))
    candidates = [row for row in rows if row['free_mib'] >= minimum_mib]
    if not candidates:
        raise RuntimeError(f'No GPU has required free memory ({minimum_mib} MiB): {rows}')
    return max(candidates, key=lambda row: row['free_mib'])
