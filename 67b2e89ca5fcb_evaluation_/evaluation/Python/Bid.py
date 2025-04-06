import os
import time
import joblib
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from Bidder import Bidder
from BidRequest import BidRequest
from numba import jit
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class Bid(Bidder):
    def __init__(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        self.ctr_model = self._optimize_model(joblib.load(os.path.join(base_path, 'ctr_model.pkl')))
        self.cvr_model = self._optimize_model(joblib.load(os.path.join(base_path, 'cvr_model.pkl')))
        self.market_price_model = self._optimize_model(joblib.load(os.path.join(base_path, 'market_price_model.pkl')))
        label_encoders = joblib.load(os.path.join(base_path, 'label_encoders.pkl'))
        self.label_maps = {
            col: {str(val): idx for idx, val in enumerate(encoder.classes_)}
            for col, encoder in label_encoders.items()
        }
        self.scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        self.scale_mean = self.scaler.mean_ if self.scaler else None
        self.scale_std = self.scaler.scale_ if self.scaler else None
        self.advertiser_weights = {
            '3476': 10,
            '1458': 0,
            '3358': 2,
            '3386': 0,
            '3427': 0
        }
        self.total_inference_time = 0
        self.num_requests = 0
        self.max_inference_time = 0
        self.min_inference_time = float('inf')

    def _optimize_model(self, model):
        if hasattr(model, 'n_jobs'):
            model.n_jobs = -1
        return model

    @staticmethod
    @jit(nopython=True)
    def _fast_numeric_scaling(values, mean, std):
        return (values - mean) / std

    def _get_hour_and_day(self, timestamp):
        try:
            ts = int(timestamp)
            hour = int(str(ts)[8:10])
            day = datetime.strptime(str(ts)[:8], '%Y%m%d').weekday()
            is_weekend = float(day >= 5)
            return hour, day, is_weekend
        except:
            return -1, -1, -1

    def _extract_features(self, bid_request):
        features = np.zeros(23, dtype=np.float32)
        hour, day, is_weekend = self._get_hour_and_day(bid_request.timestamp)
        features[0:3] = [hour, day, is_weekend]
        
        try:
            width = float(bid_request.ad_slot_width)  # Changed from adSlotWidth
            height = float(bid_request.ad_slot_height)  # Changed from adSlotHeight
            area = width * height
            features[3:8] = [
                width,
                height,
                area,
                float(area >= 100000),
                float(bid_request.ad_slot_floor_price or 0)  # Changed from adSlotFloorPrice
            ]
        except:
            features[3:8] = 0

        ua = bid_request.user_agent or ''  # Changed from userAgent
        features[9:13] = [
            float('Mobile' in ua or 'Android' in ua or 'iOS' in ua),
            float('Chrome' in ua),
            float('Firefox' in ua),
            float('Safari' in ua)
        ]

        categorical_values = [
            ('Region', bid_request.region),
            ('City', bid_request.city),
            ('Adexchange', bid_request.ad_exchange),  # Changed from adExchange
            ('Domain', bid_request.domain),
            ('URL', bid_request.url),
            ('AdslotID', bid_request.ad_slot_id),  # Changed from adSlotID
            ('Adslotvisibility', bid_request.ad_slot_visibility),  # Changed from adSlotVisibility
            ('Adslotformat', bid_request.ad_slot_format),  # Changed from adSlotFormat
            ('CreativeID', bid_request.creative_id),  # Changed from creativeID
            ('AdvertiserID', bid_request.advertiser_id)  # Changed from advertiserId
        ]

        for idx, (col, val) in enumerate(categorical_values):
            val_str = str(val if val is not None else 'unknown')
            features[13 + idx] = self.label_maps[col].get(val_str, 0)

        if self.scaler is not None:
            features[3:7] = self._fast_numeric_scaling(
                features[3:7],
                self.scale_mean,
                self.scale_std
            )
        return features.reshape(1, -1)

    def get_bid_price(self, bid_request: BidRequest) -> int:
        start_time = time.time()
        try:
            features = self._extract_features(bid_request)
            ctr = self.ctr_model.predict_proba(features)[0, 1]
            cvr = self.cvr_model.predict_proba(features)[0, 1]
            market_price = self.market_price_model.predict(features)[0]
            
            conversion_weight = self.advertiser_weights.get(str(bid_request.advertiser_id), 1)
            ev = ctr + conversion_weight * cvr
            
            # Default bid price calculation
            bid_price = -1 if ev < 0.5  else max(270, market_price*5)
            
            inference_time = (time.time() - start_time) * 1000
            self.total_inference_time += inference_time
            self.num_requests += 1
            self.max_inference_time = max(self.max_inference_time, inference_time)
            self.min_inference_time = min(self.min_inference_time, inference_time)
            
            return int(round(bid_price))
        except Exception as e:
            print(f"Error in getBidPrice: {str(e)}")
            return -1