# uav_api_client.py
"""
UAV API Client
Wrapper for the UAV Control System API to simplify drone operations
"""
import requests
from typing import Dict, List, Any, Optional


class UAVAPIClient:
    """Client for interacting with the UAV Control System API"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize UAV API Client

        Args:
            base_url: Base URL of the UAV API server
            api_key: Optional API key for authentication (defaults to USER role if not provided)
                    - None or empty: USER role (basic access)
                    - Valid key: SYSTEM or ADMIN role (based on key)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        if self.api_key:
            self.headers['X-API-Key'] = self.api_key

    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"

        # Merge authentication headers with any provided headers
        headers = kwargs.pop('headers', {})
        headers.update(self.headers)

        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            if response.status_code == 204:
                return None
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise Exception(f"Authentication failed: Invalid API key")
            elif e.response.status_code == 403:
                error_detail = e.response.json().get('detail', 'Access denied')
                raise Exception(f"Permission denied: {error_detail}")
            raise Exception(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    # Drone Operations
    def list_drones(self) -> List[Dict[str, Any]]:
        """Get all drones in the current session"""
        return self._request('GET', '/drones')

    def get_drone_status(self, drone_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific drone"""
        return self._request('GET', f'/drones/{drone_id}')

    def take_off(self, drone_id: str, altitude: float = 10.0) -> Dict[str, Any]:
        """Command drone to take off to specified altitude"""
        return self._request('POST', f'/drones/{drone_id}/command/take_off',params={'altitude': altitude})

    def land(self, drone_id: str) -> Dict[str, Any]:
        """Command drone to land at current position"""
        return self._request('POST', f'/drones/{drone_id}/command/land')

    def move_to(self, drone_id: str, x: float, y: float, z: float) -> Dict[str, Any]:
        """Move drone to specific coordinates"""
        return self._request('POST', f'/drones/{drone_id}/command/move_to',
                            params={'x': x, 'y': y, 'z': z})

    def move_along_path(self, drone_id: str, waypoints: List[Dict[str, float]]) -> Dict[str, Any]:
        """Move drone along a path of waypoints"""
        return self._request('POST', f'/drones/{drone_id}/command/move_along_path',
                            json={'waypoints': waypoints})

    def change_altitude(self, drone_id: str, altitude: float) -> Dict[str, Any]:
        """Change drone altitude while maintaining X/Y position"""
        return self._request('POST', f'/drones/{drone_id}/command/change_altitude',
                            params={'altitude': altitude})

    def hover(self, drone_id: str, duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Command drone to hover at current position.
        
        Args:
            drone_id: ID of the drone
            duration: Optional duration to hover in seconds
        """
        params = {}
        if duration is not None:
            params['duration'] = duration
        return self._request('POST', f'/drones/{drone_id}/command/hover', params=params)

    def rotate(self, drone_id: str, heading: float) -> Dict[str, Any]:
        """Rotate drone to face specific direction (0-360 degrees)"""
        return self._request('POST', f'/drones/{drone_id}/command/rotate',
                            params={'heading': heading})
    
    def move_towards(self, drone_id: str, distance: float, heading: Optional[float] = None, 
                    dz: Optional[float] = None) -> Dict[str, Any]:
        """
        Move drone a specific distance in a direction.
        
        Args:
            drone_id: ID of the drone
            distance: Distance to move in meters
            heading: Optional heading direction (0-360). If None, uses current heading.
            dz: Optional vertical component (altitude change)
        """
        params = {'distance': distance}
        if heading is not None:
            params['heading'] = heading
        if dz is not None:
            params['dz'] = dz
        return self._request('POST', f'/drones/{drone_id}/command/move_towards', params=params)

    def return_home(self, drone_id: str) -> Dict[str, Any]:
        """Command drone to return to home position"""
        return self._request('POST', f'/drones/{drone_id}/command/return_home')
        
    def set_home(self, drone_id: str) -> Dict[str, Any]:
        """Set current position as home position"""
        return self._request('POST', f'/drones/{drone_id}/command/set_home')
        
    def calibrate(self, drone_id: str) -> Dict[str, Any]:
        """Calibrate drone sensors"""
        return self._request('POST', f'/drones/{drone_id}/command/calibrate')

    def charge(self, drone_id: str, charge_amount: float) -> Dict[str, Any]:
        """Charge drone battery (when landed)"""
        return self._request('POST', f'/drones/{drone_id}/command/charge',
                            params={'charge_amount': charge_amount})

    def take_photo(self, drone_id: str) -> Dict[str, Any]:
        """Take a photo with drone camera"""
        return self._request('POST', f'/drones/{drone_id}/command/take_photo')
        
    def send_message(self, drone_id: str, target_drone_id: str, message: str) -> Dict[str, Any]:
        """
        Send a message to another drone.
        
        Args:
            drone_id: ID of the sender drone
            target_drone_id: ID of the recipient drone
            message: Content of the message
        """
        return self._request('POST', f'/drones/{drone_id}/command/send_message', 
                           params={'target_drone_id': target_drone_id, 'message': message})
                           
    def broadcast(self, drone_id: str, message: str) -> Dict[str, Any]:
        """
        Broadcast a message to all other drones.
        
        Args:
            drone_id: ID of the sender drone
            message: Content of the message
        """
        return self._request('POST', f'/drones/{drone_id}/command/broadcast', 
                           params={'message': message})

    # Session Operations
    def get_current_session(self) -> Dict[str, Any]:
        """Get information about current mission session"""
        return self._request('GET', '/sessions/current')

    def get_session_data(self, session_id: str = 'current') -> Dict[str, Any]:
        """Get all entities in a session (drones, targets, obstacles, environment)"""
        return self._request('GET', f'/sessions/{session_id}/data')

    def get_task_progress(self, session_id: str = 'current') -> Dict[str, Any]:
        """Get mission task completion progress"""
        return self._request('GET', f'/sessions/{session_id}/task-progress')

    # Environment Operations
    def get_weather(self) -> Dict[str, Any]:
        """Get current weather conditions"""
        return self._request('GET', '/environments/current')

    def get_targets(self) -> List[Dict[str, Any]]:
        """Get all targets in the session"""
        return self._request('GET', '/targets')

    def get_waypoints(self) -> List[Dict[str, Any]]:
        """Get all charging station waypoints"""
        return self._request('GET', '/targets/waypoints')

    def get_obstacles(self) -> List[Dict[str, Any]]:
        """Get all obstacles in the session"""
        return self._request('GET', '/obstacles')

    def get_nearby_entities(self, drone_id: str) -> Dict[str, Any]:
        """Get entities near a drone (within perceived radius)"""
        return self._request('GET', f'/drones/{drone_id}/nearby')

    # Safety Operations
    def check_point_collision(self, x: float, y: float, z: float,
                             safety_margin: float = 0.0) -> Optional[Dict[str, Any]]:
        """Check if a point collides with any obstacle"""
        result = self._request('POST', '/obstacles/collision/check',
                              json={
                                  'point': {'x': x, 'y': y, 'z': z},
                                  'safety_margin': safety_margin
                              })
        return result

    def check_path_collision(self, start_x: float, start_y: float, start_z: float,
                            end_x: float, end_y: float, end_z: float,
                            safety_margin: float = 1.0) -> Optional[Dict[str, Any]]:
        """Check if a path intersects any obstacle"""
        result = self._request('POST', '/obstacles/collision/path',
                              json={
                                  'start': {'x': start_x, 'y': start_y, 'z': start_z},
                                  'end': {'x': end_x, 'y': end_y, 'z': end_z},
                                  'safety_margin': safety_margin
                              })
        return result
