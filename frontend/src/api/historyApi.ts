import API from "./authApi";
import type { PredictionResult } from "../utils/PredictionModel";

export const savePredictionHistory = (prediction: PredictionResult) => {
    return API.post("/prediction-history", prediction);
};

export const getPredictionHistory = () => {
    return API.get("/prediction-history");
};