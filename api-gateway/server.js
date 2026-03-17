const express = require("express");
const logger = require("./middleware/logger");
const auth = require("./middleware/auth");

const predictRoute = require("./routes/predict");

const app = express();

app.use(express.json());
app.use(logger);

// apply auth (for now placeholder)
app.use(auth);

app.use("/predict", predictRoute);

app.get("/", (req, res) => {
    res.send("API Gateway Running");
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Gateway running on port ${PORT}`);
});