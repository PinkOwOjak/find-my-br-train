# live.py - Live train viewing and health check endpoints

from fastapi.responses import HTMLResponse
from typing import Dict, Any
import time


def health_check(current_revision: int, tracker):
    """Health check endpoint"""
    redis_healthy = tracker.health_check()
    active_trains = tracker.get_active_train_count()
    trains_with_history = len(tracker.get_all_trains_with_history())
    
    return {
        "status": "healthy" if redis_healthy else "degraded",
        "timestamp": int(time.time()),
        "revision": current_revision,
        "redis_connected": redis_healthy,
        "active_trains": active_trains,
        "trains_with_history": trains_with_history
    }


def get_active_trains_details(tracker, include_plan: bool = False) -> Dict:
    """Return JSON details of currently active trains. The 10-minute motion
    plan is opt-in (?plan=1) so older apps polling every 10 s don't pay for it."""
    active_trains = tracker.get_all_active_trains()
    positions = tracker.get_positions(active_trains)
    
    result = {}
    for tid, info in positions.items():
        if info.get("is_live"):
            result[tid] = {
                "position": info.get("position"),
                "timestamp": info.get("timestamp"),
                "active_user": info.get("active_user", 0),
                "is_live": True,
                "speed_kmh": info.get("speed_kmh"),
                "moving": info.get("moving", False),
            }
            if include_plan:
                result[tid]["plan"] = info.get("plan")
    return result


def view_live_trains(tracker, data: Dict[str, Any]):
    """View all trains with position data (live + historical)"""
    try:
        # Get all trains with any position history (up to 10 hours)
        all_trains = tracker.get_all_trains_with_history()
        active_trains = set(tracker.get_all_active_trains())
        
        # Get positions for all trains
        positions = tracker.get_positions(all_trains)
        
        # Sort by: live trains first, then by timestamp (newest first)
        all_trains = sorted(
            all_trains,
            key=lambda tid: (
                0 if tid in active_trains else 1,  # Live trains first
                -positions.get(tid, {}).get("timestamp", 0)  # Then by timestamp
            )
        )
        
        tid_to_name = data.get("train_names", {})
        
        total_trains = len(all_trains)
        live_count = len(active_trains)
        historical_count = total_trains - live_count
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Find My BR Train - Live Trains</title>
            <meta charset="UTF-8">
            <style>
                .live {{ border-left: 4px solid #22c55e; }}
                .historical {{ border-left: 4px solid #f59e0b; opacity: 0.8; }}
            </style>
        </head>
        <body>
            <h1>🚂 Find My BR Train - Live Trains</h1>
            
            <h2>📊 Statistics</h2>
            <ul>
                <li><strong>🟢 Live Trains:</strong> {live_count} (active in last 10 min)</li>
                <li><strong>🟡 Historical:</strong> {historical_count} (last known position, up to 10h)</li>
                <li><strong>Total:</strong> {total_trains}</li>
                <li><strong>Redis Status:</strong> {"✅ Connected" if tracker.health_check() else "❌ Disconnected"}</li>
            </ul>
            
            <h2>🚂 All Trains ({total_trains} total)</h2>
        """
        
        if not all_trains:
            html_content += "<p><em>No trains with position data.</em></p>"
        else:
            html_content += "<div>"
            
            for i, train_id in enumerate(all_trains):
                train_name = tid_to_name.get(train_id, "Unknown Train")
                is_live = train_id in active_trains
                
                position_data = positions.get(train_id)
                if position_data:
                    position = position_data["position"]
                    timestamp = position_data["timestamp"]
                    user_count = position_data.get("active_user", 0)
                    is_live_data = position_data.get("is_live", False)
                    
                    position_info = f"Position: {position:.2f}"
                    
                    age_seconds = int(time.time() - timestamp)
                    if age_seconds < 60:
                        timestamp_info = f"Updated: {age_seconds}s ago"
                    elif age_seconds < 3600:
                        timestamp_info = f"Updated: {age_seconds // 60}m ago"
                    else:
                        timestamp_info = f"Updated: {age_seconds // 3600}h {(age_seconds % 3600) // 60}m ago"
                    
                    if is_live_data:
                        status = f"🟢 Live | Users: {user_count}"
                        css_class = "live"
                    else:
                        status = "🟡 Last Known Position"
                        css_class = "historical"
                else:
                    position_info = "Position: Unknown"
                    timestamp_info = ""
                    status = "❓ Unknown"
                    css_class = "historical"
                
                html_content += f"""
                <div class="{css_class}" style="border: 1px solid #ddd; margin: 5px 0; padding: 10px;">
                    <h3>#{i+1}: {train_name}</h3>
                    <p><strong>{position_info}</strong> | {status}</p>
                    <p><small>{timestamp_info}</small></p>
                </div>
                """
            
            html_content += "</div>"
        
        html_content += f"""
            <hr>
            <p><small>Last updated: {int(time.time())} | <a href="/live">Refresh</a> | <a href="/health">Health Check</a></small></p>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        return HTMLResponse(content=f"""
        <html>
        <body>
            <h1>Error Loading Live Trains</h1>
            <p>Error: {str(e)}</p>
        </body>
        </html>
        """)
