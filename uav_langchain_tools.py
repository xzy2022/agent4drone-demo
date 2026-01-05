"""
LangChain Tools for UAV Control
Wraps the UAV API client as LangChain tools using @tool decorator
All tools accept JSON string input for consistent parameter handling
"""
from langchain.tools import tool
from uav_api_client import UAVAPIClient
import json


def create_uav_tools(client: UAVAPIClient) -> list:
    """
    Create all UAV control tools for LangChain agent using @tool decorator
    All tools that require parameters accept a JSON string input
    """

    # ========== Information Gathering Tools (No Parameters) ==========

    @tool
    def list_drones() -> str:
        """List all available drones in the current session with their status, battery level, and position.
        Use this to see what drones are available before trying to control them.

        No input required."""
        try:
            drones = client.list_drones()
            return json.dumps(drones, indent=2)
        except Exception as e:
            return f"Error listing drones: {str(e)}"

    @tool
    def get_session_info() -> str:
        """Get current session information including task type, statistics, and status.
        Use this to understand what mission you need to complete.

        No input required."""
        try:
            session = client.get_current_session()
            return json.dumps(session, indent=2)
        except Exception as e:
            return f"Error getting session info: {str(e)}"

    @tool
    def get_task_progress() -> str:
        """Get mission task progress including completion percentage and status.
        Use this to track mission completion and see how close you are to finishing.

        No input required."""
        try:
            progress = client.get_task_progress()
            return json.dumps(progress, indent=2)
        except Exception as e:
            return f"Error getting task progress: {str(e)}"

    @tool
    def get_weather() -> str:
        """Get current weather conditions including wind speed, visibility, and weather type.
        Check this before takeoff to ensure safe flying conditions.

        No input required."""
        try:
            weather = client.get_weather()
            return json.dumps(weather, indent=2)
        except Exception as e:
            return f"Error getting weather: {str(e)}"

    # @tool
    # def get_targets() -> str:
    #     """Get all targets in the session including waypoints, survey points, and areas to search or patrol.
    #     Use this to see what targets you need to visit.

    #     No input required."""
    #     try:
    #         targets = client.get_targets()
    #         return json.dumps(targets, indent=2)
    #     except Exception as e:
    #         return f"Error getting targets: {str(e)}"

    # @tool
    # def get_obstacles() -> str:
    #     """Get all obstacles in the session that drones must avoid.
    #     Use this to understand what obstacles exist in the environment.

    #     No input required."""
    #     try:
    #         obstacles = client.get_obstacles()
    #         return json.dumps(obstacles, indent=2)
    #     except Exception as e:
    #         return f"Error getting obstacles: {str(e)}"


    @tool
    def get_drone_status(input_json: str) -> str:
        """Get detailed status of a specific drone including position, battery, heading, and visited targets.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)

        Example: {{"drone_id": "drone-001"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')

            if not drone_id:
                return "Error: drone_id is required"

            status = client.get_drone_status(drone_id)
            return json.dumps(status, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error getting drone status: {str(e)}"

    @tool
    def get_nearby_entities(input_json: str) -> str:
        """Get drones, targets, and obstacles near a specific drone (within its perception radius).

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)

        Example: {{"drone_id": "drone-001"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')

            if not drone_id:
                return "Error: drone_id is required"

            nearby = client.get_nearby_entities(drone_id)
            return json.dumps(nearby, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error getting nearby entities: {str(e)}"

    @tool
    def land(input_json: str) -> str:
        """Command a drone to land at its current position.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)

        Example: {{"drone_id": "drone-001"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')

            if not drone_id:
                return "Error: drone_id is required"

            result = client.land(drone_id)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error during landing: {str(e)}"

    @tool
    def hover(input_json: str) -> str:
        """Command a drone to hover at its current position.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)
        - duration: Optional duration in seconds to hover (optional)

        Example: {{"drone_id": "drone-001", "duration": 5.0}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            duration = params.get('duration')

            if not drone_id:
                return "Error: drone_id is required"

            result = client.hover(drone_id, duration)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error hovering: {str(e)}"

    @tool
    def return_home(input_json: str) -> str:
        """Command a drone to return to its home position.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)

        Example: {{"drone_id": "drone-001"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')

            if not drone_id:
                return "Error: drone_id is required"

            result = client.return_home(drone_id)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error returning home: {str(e)}"

    @tool
    def set_home(input_json: str) -> str:
        """Set the drone's current position as its new home position.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)

        Example: {{"drone_id": "drone-001"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')

            if not drone_id:
                return "Error: drone_id is required"

            result = client.set_home(drone_id)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error setting home: {str(e)}"

    @tool
    def calibrate(input_json: str) -> str:
        """Calibrate the drone's sensors.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)

        Example: {{"drone_id": "drone-001"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')

            if not drone_id:
                return "Error: drone_id is required"

            result = client.calibrate(drone_id)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error calibrating: {str(e)}"

    @tool
    def take_photo(input_json: str) -> str:
        """Command a drone to take a photo.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)

        Example: {{"drone_id": "drone-001"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')

            if not drone_id:
                return "Error: drone_id is required"

            result = client.take_photo(drone_id)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\"}}"
        except Exception as e:
            return f"Error taking photo: {str(e)}"

    # ========== Two Parameter Tools ==========

    @tool
    def take_off(input_json: str) -> str:
        """Command a drone to take off to a specified altitude.
        Drone must be on ground (idle or ready status).

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)
        - altitude: Target altitude in meters (optional, default: 10.0)

        Example: {{"drone_id": "drone-001", "altitude": 15.0}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            altitude = params.get('altitude', 10.0)

            if not drone_id:
                return "Error: drone_id is required"

            result = client.take_off(drone_id, altitude)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"altitude\": 15.0}}"
        except Exception as e:
            return f"Error during takeoff: {str(e)}"

    @tool
    def change_altitude(input_json: str) -> str:
        """Change a drone's altitude while maintaining X/Y position.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)
        - altitude: Target altitude in meters (required)

        Example: {{"drone_id": "drone-001", "altitude": 20.0}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            altitude = params.get('altitude')

            if not drone_id:
                return "Error: drone_id is required"
            if altitude is None:
                return "Error: altitude is required"

            result = client.change_altitude(drone_id, altitude)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"altitude\": 20.0}}"
        except Exception as e:
            return f"Error changing altitude: {str(e)}"

    @tool
    def rotate(input_json: str) -> str:
        """Rotate a drone to face a specific direction.
        0=North, 90=East, 180=South, 270=West.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)
        - heading: Target heading in degrees 0-360 (required)

        Example: {{"drone_id": "drone-001", "heading": 90.0}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            heading = params.get('heading')

            if not drone_id:
                return "Error: drone_id is required"
            if heading is None:
                return "Error: heading is required"

            result = client.rotate(drone_id, heading)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"heading\": 90.0}}"
        except Exception as e:
            return f"Error rotating: {str(e)}"

    @tool
    def send_message(input_json: str) -> str:
        """Send a message from one drone to another.

        Input should be a JSON string with:
        - drone_id: The ID of the sender drone (required)
        - target_drone_id: The ID of the recipient drone (required)
        - message: The message content (required)

        Example: {{"drone_id": "drone-001", "target_drone_id": "drone-002", "message": "Hello"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            target_drone_id = params.get('target_drone_id')
            message = params.get('message')

            if not drone_id:
                return "Error: drone_id is required"
            if not target_drone_id:
                return "Error: target_drone_id is required"
            if not message:
                return "Error: message is required"

            result = client.send_message(drone_id, target_drone_id, message)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"target_drone_id\": \"drone-002\", \"message\": \"...\"}}"
        except Exception as e:
            return f"Error sending message: {str(e)}"

    @tool
    def broadcast(input_json: str) -> str:
        """Broadcast a message from one drone to all other drones.

        Input should be a JSON string with:
        - drone_id: The ID of the sender drone (required)
        - message: The message content (required)

        Example: {{"drone_id": "drone-001", "message": "Alert"}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            message = params.get('message')

            if not drone_id:
                return "Error: drone_id is required"
            if not message:
                return "Error: message is required"

            result = client.broadcast(drone_id, message)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"message\": \"...\"}}"
        except Exception as e:
            return f"Error broadcasting: {str(e)}"

    @tool
    def charge(input_json: str) -> str:
        """Command a drone to charge its battery.
        Drone must be landed at a charging station.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)
        - charge_amount: Amount to charge in percent (required)

        Example: {{"drone_id": "drone-001", "charge_amount": 25.0}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            charge_amount = params.get('charge_amount')

            if not drone_id:
                return "Error: drone_id is required"
            if charge_amount is None:
                return "Error: charge_amount is required"

            result = client.charge(drone_id, charge_amount)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"charge_amount\": 25.0}}"
        except Exception as e:
            return f"Error charging: {str(e)}"

    @tool
    def move_towards(input_json: str) -> str:
        """Move a drone a specific distance in a direction.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)
        - distance: Distance to move in meters (required)
        - heading: Heading direction in degrees 0-360 (optional, default: current heading)
        - dz: Vertical component in meters (optional)

        Example: {{"drone_id": "drone-001", "distance": 10.0, "heading": 90.0}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            distance = params.get('distance')
            heading = params.get('heading')
            dz = params.get('dz')

            if not drone_id:
                return "Error: drone_id is required"
            if distance is None:
                return "Error: distance is required"

            result = client.move_towards(drone_id, distance, heading, dz)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"distance\": 10.0}}"
        except Exception as e:
            return f"Error moving towards: {str(e)}"

    # @tool
    # def move_along_path(input_json: str) -> str:
    #     """Move a drone along a path of waypoints.

    #     Input should be a JSON string with:
    #     - drone_id: The ID of the drone (required)
    #     - waypoints: List of points with x, y, z coordinates (required)

    #     Example: {{"drone_id": "drone-001", "waypoints": [{{"x": 10, "y": 10, "z": 10}}, {{"x": 20, "y": 20, "z": 10}}]}}
    #     """
    #     try:
    #         params = json.loads(input_json) if isinstance(input_json, str) else input_json
    #         drone_id = params.get('drone_id')
    #         waypoints = params.get('waypoints')

    #         if not drone_id:
    #             return "Error: drone_id is required"
    #         if not waypoints:
    #             return "Error: waypoints list is required"

    #         result = client.move_along_path(drone_id, waypoints)
    #         return json.dumps(result, indent=2)
    #     except json.JSONDecodeError as e:
    #         return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"waypoints\": [...]}}"
    #     except Exception as e:
    #         return f"Error moving along path: {str(e)}"

    # ========== Multi-Parameter Tools ==========

    @tool
    def move_to(input_json: str) -> str:
        """Move a drone to specific 3D coordinates (x, y, z).
        Always check for collisions first using check_path_collision.

        Input should be a JSON string with:
        - drone_id: The ID of the drone (required)
        - x: Target X coordinate in meters (required)
        - y: Target Y coordinate in meters (required)
        - z: Target Z coordinate (altitude) in meters (required)

        Example: {{"drone_id": "drone-001", "x": 100.0, "y": 50.0, "z": 20.0}}
        """
        try:
            params = json.loads(input_json) if isinstance(input_json, str) else input_json
            drone_id = params.get('drone_id')
            x = params.get('x')
            y = params.get('y')
            z = params.get('z')

            if not drone_id:
                return "Error: drone_id is required"
            if x is None or y is None or z is None:
                return "Error: x, y, and z coordinates are required"

            result = client.move_to(drone_id, x, y, z)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Expected format: {{\"drone_id\": \"drone-001\", \"x\": 100.0, \"y\": 50.0, \"z\": 20.0}}"
        except Exception as e:
            return f"Error moving drone: {str(e)}"


    # Return all tools
    return [
        list_drones,
        get_drone_status,
        get_session_info,
        get_task_progress,
        get_weather,
        get_nearby_entities,
        take_off,
        land,
        move_to,
        move_towards,
        change_altitude,
        hover,
        rotate,
        return_home,
        set_home,
        calibrate,
        take_photo,
        send_message,
        broadcast,
        charge,
    ]
