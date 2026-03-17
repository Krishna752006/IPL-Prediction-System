const express = require("express");
const router = express.Router();
const axios = require("axios");

router.post("/", async (req, res) => {
    try {
        const response = await axios.post(
            "http://ml-service:5000/predict",
            req.body
        );

        res.json(response.data);
    } catch (err) {
        console.error(err.message);
        res.status(500).json({ error: "ML service failed" });
    }
});

module.exports = router;