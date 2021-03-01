evalscript_cloud_mask = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: [
                    "CLD",
                    "dataMask"
                ]
            }],
            output: {
                bands: 1,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        if (sample.dataMask == 1){
            if (sample.CLD > 0){
                return [2]
            }
            return [1]
        }
        if (sample.dataMask == 0){
            return [0]
        }
    }
"""

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "dataMask"]
            }],
            output: {
                bands: 4
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02, sample.dataMask];
    }
"""