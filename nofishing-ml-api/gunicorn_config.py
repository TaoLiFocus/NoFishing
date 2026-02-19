"""
Gunicorn Configuration for NoFishing ML API
"""
import multiprocessing
import os

# Socket
bind = "0.0.0.0:5000"
backlog = 2048

# Workers
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "nofishing-ml-api"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
umask = 0o007
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None
ssl_version = None
cert_reqs = None
ca_certs = None
suppress_ragged_eofs = True
do_handshake_on_connect = False

# Server hooks
def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def pre_fork(server, worker):
    pass

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("NoFishing ML API server is ready. Spawning workers")

def pre_request(worker, req):
    worker.log.debug(f"{req.method} {req.path}")

def post_request(worker, req, environ, resp):
    pass

def child_exit(server, worker):
    server.log.info(f"Worker died (pid: {worker.pid})")

def worker_abort(worker):
    server.log.info(f"Worker received SIGABRT signal (pid: {worker.pid})")

def nworkers_changed(server, new_value, old_value):
    server.log.info(f"Worker count changed: {old_value} -> {new_value}")
