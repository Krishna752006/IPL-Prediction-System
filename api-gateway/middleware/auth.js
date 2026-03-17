module.exports = (req, res, next) => {
    console.log("Auth middleware hit (placeholder)");
    
    // future: verify JWT here
    next();
};