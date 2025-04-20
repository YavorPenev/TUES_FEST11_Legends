import React from "react";
import { Link } from "react-router";


function UsefulSources() {
    return (
        <div className="mt-5 w-full max-w-screen-lg mx-auto px-4">
            <Link
                to="/UsefulSources"
                className="cursor-pointer rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition transform hover:scale-105 bg-blue-900 text-white flex flex-col md:flex-row items-center p-6"
            >
                <img
                    src="./source.png"
                    alt="Investments and Finance"
                    className="w-full md:w-1/2 h-48 md:h-64 object-cover rounded-3xl mb-4 md:mb-0 md:mr-6"
                />

                <div className="flex-1">
                    <h2 className="text-xl md:text-2xl font-bold mb-2">
                        Explore Useful Investment Resources          </h2>
                    <p className="text-sm md:text-base text-left indent-4">
                        Discover a curated list of trusted websites and tools that provide
                        valuable insights into stock data, investment applications, and
                        strategies for making informed financial decisions. Whether you're
                        looking for real-time market updates, portfolio management tools, or
                        expert advice, these resources will help you navigate the world of
                        investments with confidence.{" "}
                        <b className="text-1xl">Click here</b> to access the list and start exploring!
                    </p>
                </div>
            </Link>
        </div>
    );
};

export default UsefulSources;