#!/bin/bash

source .env
: "${SERVER_HOST:?Please set SERVER_HOST in environment.sh}"
: "${CRAWL_PAGE_PORT:?Please set CRAWL_PAGE_PORT in environment.sh}"
: "${WEBSEARCH_PORT:?Please set WEBSEARCH_PORT in environment.sh}"
: "${CODE_EXEC_PORT:?Please set CODE_EXEC_PORT in environment.sh}"
: "${NSJAILPATH:?Please set NSJAILPATH in environment.sh}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DIR/logs/$SERVER_HOST";   mkdir -p "$LOG_DIR"
PID_DIR="$DIR/pids/$SERVER_HOST";   mkdir -p "$PID_DIR"

cmd=$1
if [[ ! "$cmd" =~ ^(start|stop|status|test)$ ]]; then
  echo "Usage: $0 [start|stop|status|test]"
  exit 1
fi

# ---------------------------------------------
#                start
# ---------------------------------------------
if [[ "$cmd" == "start" ]]; then
  # CrawlPage
  pidf="$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid"
  logf="$LOG_DIR/CrawlPage_$CRAWL_PAGE_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "CrawlPage is already running (PID $(cat "$pidf"))"
  else
    echo "Starting CrawlPage on port $SERVER_HOST:$CRAWL_PAGE_PORT..."
    nohup python -u "$DIR/version/crawl_page_server.py" > "$logf" 2>&1 &
    echo $! > "$pidf"
  fi

  # WebSearch
  pidf="$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid"
  logf="$LOG_DIR/WebSearch_$WEBSEARCH_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "WebSearch is already running (PID $(cat "$pidf"))"
  else
    echo "Starting WebSearch on port $SERVER_HOST:$WEBSEARCH_PORT..."
    nohup python -u "$DIR/version/cache_serper_server.py" > "$logf" 2>&1 &
    echo $! > "$pidf"
  fi

  # CodeExec
  pidf="$PID_DIR/${SERVER_HOST}_CodeExec_$CODE_EXEC_PORT.pid"
  logf="$LOG_DIR/CodeExec_$CODE_EXEC_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "CodeExec is already running (PID $(cat "$pidf"))"
  else
    echo "Starting CodeExec on port $SERVER_HOST:$CODE_EXEC_PORT..."
    nohup python -u "$DIR/version/code_execute_server.py" > "$logf" 2>&1 &
    echo $! > "$pidf"
  fi

# ---------------------------------------------
#                test
# ---------------------------------------------
elif [[ "$cmd" == "test" ]]; then
  echo "--------------------Setting Test Environment Variables------------------"
  source "$DIR/version/TestEnvironment.sh"
  echo "--------------------Setup Successful------------------"

  echo "--------------------Starting web search test------------------"
  python -u "$DIR/version/test_cache_serper_server.py" \
          "http://$SERVER_HOST:$WEBSEARCH_PORT/search"
  echo "-------------------------Test Completed--------------------------"
  echo "--------------------Starting crawl page test-------------------"
  python -u "$DIR/version/test_crawl_page_simple.py" \
          "http://$SERVER_HOST:$CRAWL_PAGE_PORT/crawl_page"
  echo "-------------------------Test Completed--------------------------"

  echo "--------------------Starting code exec test-------------------"
  python -u "$DIR/version/test_code_execute_server.py" \
          "http://$SERVER_HOST:$CODE_EXEC_PORT/code_exec"
  echo "-------------------------Test Completed--------------------------"

# ---------------------------------------------
#                stop
# ---------------------------------------------
elif [[ "$cmd" == "stop" ]]; then
  # CrawlPage
  pidf="$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid"
  stopped=0
  
  # First try to stop using PID file
  if [[ -f "$pidf" ]]; then
    pid=$(cat "$pidf" 2>/dev/null)
    if [[ -n "$pid" ]]; then
      if kill -0 "$pid" 2>/dev/null; then
        if kill "$pid" 2>/dev/null; then
          echo "CrawlPage stopped (PID $pid)"
          stopped=1
        else
          echo "Warning: Unable to stop CrawlPage with PID $pid, trying other methods..."
        fi
      fi
    fi
    # Delete PID file regardless of success
    rm -f "$pidf"
  fi
  
  # Check and stop all processes on the port
  port_processes=($(lsof -t -i:"$CRAWL_PAGE_PORT" 2>/dev/null))
  if [[ ${#port_processes[@]} -gt 0 ]]; then
    for p in "${port_processes[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        if kill "$p" 2>/dev/null; then
          echo "Stopped CrawlPage via port (PID $p)"
          stopped=1
        else
          echo "Warning: Unable to stop process $p on port $CRAWL_PAGE_PORT"
        fi
      fi
    done
  fi
  
  if [[ $stopped -eq 0 ]]; then
    echo "CrawlPage is not running, and port $CRAWL_PAGE_PORT is not in use"
  fi

  # WebSearch
  pidf="$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid"
  stopped=0
  
  # First try to stop using PID file
  if [[ -f "$pidf" ]]; then
    pid=$(cat "$pidf" 2>/dev/null)
    if [[ -n "$pid" ]]; then
      if kill -0 "$pid" 2>/dev/null; then
        if kill "$pid" 2>/dev/null; then
          echo "WebSearch stopped (PID $pid)"
          stopped=1
        else
          echo "Warning: Unable to stop WebSearch with PID $pid, trying other methods..."
        fi
      fi
    fi
    # Delete PID file regardless of success
    rm -f "$pidf"
  fi
  
  # Check and stop all processes on the port
  port_processes=($(lsof -t -i:"$WEBSEARCH_PORT" 2>/dev/null))
  if [[ ${#port_processes[@]} -gt 0 ]]; then
    for p in "${port_processes[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        if kill "$p" 2>/dev/null; then
          echo "Stopped WebSearch via port (PID $p)"
          stopped=1
        else
          echo "Warning: Unable to stop process $p on port $WEBSEARCH_PORT"
        fi
      fi
    done
  fi
  
  if [[ $stopped -eq 0 ]]; then
    echo "WebSearch is not running, and port $WEBSEARCH_PORT is not in use"
  fi

  # CodeExec
  pidf="$PID_DIR/${SERVER_HOST}_CodeExec_$CODE_EXEC_PORT.pid"
  stopped=0
  
  # First try to stop using PID file
  if [[ -f "$pidf" ]]; then
    pid=$(cat "$pidf" 2>/dev/null)
    if [[ -n "$pid" ]]; then
      if kill -0 "$pid" 2>/dev/null; then
        if kill "$pid" 2>/dev/null; then
          echo "CodeExec stopped (PID $pid)"
          stopped=1
        else
          echo "Warning: Unable to stop CodeExec with PID $pid, trying other methods..."
        fi
      fi
    fi
    # Delete PID file regardless of success
    rm -f "$pidf"
  fi
  
  # Check and stop all processes on the port
  port_processes=($(lsof -t -i:"$CODE_EXEC_PORT" 2>/dev/null))
  if [[ ${#port_processes[@]} -gt 0 ]]; then
    for p in "${port_processes[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        if kill "$p" 2>/dev/null; then
          echo "Stopped CodeExec via port (PID $p)"
          stopped=1
        else
          echo "Warning: Unable to stop process $p on port $CODE_EXEC_PORT"
        fi
      fi
    done
  fi
  
  if [[ $stopped -eq 0 ]]; then
    echo "CodeExec is not running, and port $CODE_EXEC_PORT is not in use"
  fi

# ---------------------------------------------
#                status
# ---------------------------------------------
else
  # CrawlPage
  pidf="$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "CrawlPage is running (PID $(cat "$pidf"))"
  elif lsof -i:"$CRAWL_PAGE_PORT" &>/dev/null; then
    echo "CrawlPage port $CRAWL_PAGE_PORT is in use, but PID file is invalid"
  else
    echo "CrawlPage is not running, and port $CRAWL_PAGE_PORT is not in use"
  fi

  # WebSearch
  pidf="$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "WebSearch is running (PID $(cat "$pidf"))"
  elif lsof -i:"$WEBSEARCH_PORT" &>/dev/null; then
    echo "WebSearch port $WEBSEARCH_PORT is in use, but PID file is invalid"
  else
    echo "WebSearch is not running, and port $WEBSEARCH_PORT is not in use"
  fi

  # CodeExec
  pidf="$PID_DIR/${SERVER_HOST}_CodeExec_$CODE_EXEC_PORT.pid"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "CodeExec is running (PID $(cat "$pidf"))"
  elif lsof -i:"$CODE_EXEC_PORT" &>/dev/null; then
    echo "CodeExec port $CODE_EXEC_PORT is in use, but PID file is invalid"
  else
    echo "CodeExec is not running, and port $CODE_EXEC_PORT is not in use"
  fi
fi
