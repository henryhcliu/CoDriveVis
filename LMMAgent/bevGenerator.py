import carla
from carla_birdeye_view import BirdViewProducer, PixelDimensions, BirdViewCropType
import cv2

class BEV_Generator:
    def __init__(self, host='localhost', port=3000, center_pos=None):
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # set the position of the spectator to be the center of the whole map
        # center for Town10HD: x=3.46, y=45.58
        if center_pos is not None:
            x_center = center_pos[0]
            y_center = center_pos[1]
        else:
            x_center = -2
            y_center = 35
        self.bev_center = [x_center, y_center]
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=self.bev_center[0], y=self.bev_center[1], z=160), carla.Rotation(pitch=-90, yaw=-90.0, roll=0.0)))
        self.world.tick()

        # generate the bird-eye view producer
        self.bev_producer = BirdViewProducer(
            self.client,  # carla.Client
            target_size=PixelDimensions(width=1900, height=1800),
            pixels_per_meter=8,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
        )
    
    def generate_bev(self, save_path=None, bev_type='schedule', free_vehicle_IDs=None, occupied_vehicle_IDs=None, passenger_IDs=None):
        if bev_type not in ['schedule', 'graph']:
            raise ValueError('The BEV type should be either "schedule" or "graph"')
        # get the bird-eye view image
        if bev_type == 'graph':
            bev = self.bev_producer.produce(predefined_loc=self.bev_center, bev_type=bev_type)
        elif bev_type == 'schedule':
            if passenger_IDs == None:
                bev = self.bev_producer.produce(predefined_loc=self.bev_center, bev_type=bev_type, free_vehicle_IDs=free_vehicle_IDs, occupied_vehicle_IDs=occupied_vehicle_IDs)
            else:
                bev = self.bev_producer.produce(predefined_loc=self.bev_center, bev_type=bev_type, free_vehicle_IDs=free_vehicle_IDs, occupied_vehicle_IDs=occupied_vehicle_IDs, unassigned_passenger_IDs=passenger_IDs)

        rgb = BirdViewProducer.as_rgb(bev) # Important function to convert the bird-eye view to RGB image based on the color map
        # transform the rgb image to BGR
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, rgb = cv2.imencode('.png', rgb)
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        if save_path is not None:
            cv2.imwrite(save_path, rgb)
        return rgb