import API from "./authApi";

export const savePredictionHistory = (prediction: any) => {
    return API.post("/prediction-history", prediction);
};

export const getPredictionHistory = () => {
    return API.get("/prediction-history");
};