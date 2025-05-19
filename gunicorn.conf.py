# gunicorn.conf.py
workers = 1  # Number of worker processes
threads = 2  # Number of threads per worker
bind = "0.0.0.0:10000"  # IP and port to bind to
worker_class = "sync"  # Worker class type
timeout = 120  # Timeout for worker processes in seconds