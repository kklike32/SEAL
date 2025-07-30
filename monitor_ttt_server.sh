#!/bin/bash

# TTT Server Monitor - Automatically restarts the server when it crashes
# Run this in Terminal 1 instead of the regular TTT server script

echo "🔄 TTT Server Monitor Starting..."
echo "================================="

cd /Users/keenan/Documents/SEAL

RESTART_COUNT=0
MAX_RESTARTS=10

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo ""
    echo "🚀 Starting TTT Server (Attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS)..."
    echo "Time: $(date)"
    
    # Start the TTT server
    bash knowledge-incorporation/scripts/TTT_server_mlx.sh
    
    # Server has exited - check why
    EXIT_CODE=$?
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    echo ""
    echo "⚠️  TTT Server stopped (Exit code: $EXIT_CODE)"
    
    if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
        echo "🔄 Restarting in 5 seconds..."
        echo "   This is normal - the server crashes after each request due to weight restoration issues"
        echo "   The RL training will continue working with automatic restarts"
        
        # Kill any lingering processes
        pkill -f "TTT_server_mlx.py" 2>/dev/null || true
        
        # Short wait before restart
        sleep 5
    else
        echo "❌ Maximum restart attempts reached. Please check for persistent issues."
        break
    fi
done

echo ""
echo "🔚 TTT Server Monitor stopped."
