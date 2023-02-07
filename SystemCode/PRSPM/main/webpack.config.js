const path = require("path");
const webpack = require("webpack");

module.exports =(env, argv)=>{
  const mode = argv.mode || 'development'
  return ({
  entry: "./ui/src/index.js",
  output: {
    path: path.resolve(__dirname, "./ui/static/main"),
    filename: "[name].js",
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
        },
      },
    ],
  },
  optimization: {
    minimize: true,
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env": {
        // This has effect on the react lib size
        NODE_ENV: JSON.stringify(mode),
      },
    }),
  ],
})
};