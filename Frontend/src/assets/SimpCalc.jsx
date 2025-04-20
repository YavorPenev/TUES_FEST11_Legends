import { useState, useRef } from "react";
import Draggable from "react-draggable"; 

function Calculator() {
  const [isVisible, setIsVisible] = useState(false); 
  const [input, setInput] = useState("");
  const nodeRef = useRef(null); //Draggable

  const handleClick = (value) => setInput((prev) => prev + value);
  const handleClear = () => setInput("");
  const handlePercent = () => {
    try {
      const result = eval(input) / 100;
      setInput(result.toString());
    } catch {
      setInput("Error");
    }
  };
  const handlePower = () => setInput((prev) => prev + "**");
  const handleResult = () => {
    try {
      const result = eval(input);
      setInput(result.toString());
    } catch {
      setInput("Error");
    }
  };

  return (
    <>
 
      {!isVisible && (
        <button
          onClick={() => setIsVisible(true)} 
          className="fixed flex bg-none bottom-4 w-18 h-18 left-4 rounded-2xl bg-gray-100 shadow-lg hover:scale-120 transition-transform hover:duration-200 active:scale-80 active:duration-50 justify-center items-center overflow-hidden"
        >
          <img
            src="/calc.png" 
            alt="Calculator"
            className="h-20 w-20"
          />
        </button>
      )}


      {isVisible && (
        <Draggable nodeRef={nodeRef}>
          <div
            ref={nodeRef}
            className="fixed bottom-20 right-4 bg-blue-800 p-4 rounded-2xl shadow-2xl w-72 border border-gray-200 cursor-grab active:cursor-grabbing"
          >
            <div className="mb-4">
              <input
                type="text"
                value={input}
                readOnly
                className="w-full p-3 border rounded text-right text-xl bg-gray-100"
              />
            </div>

            <div className="grid grid-cols-4 gap-2">
              <button
                onClick={handleClear}
                className="bg-red-500 hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50 text-white p-3 rounded-lg font-bold text-lg cursor-pointer"
              >
                C
              </button>
              <button
                onClick={handlePercent}
                className="bg-gray-300 p-3 rounded-lg font-bold text-lg cursor-pointer  hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                %
              </button>
              <button
                onClick={handlePower}
                className="bg-gray-300 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                ^
              </button>
              <button
                onClick={() => handleClick("/")}
                className="bg-blue-500 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                /
              </button>

              <button
                onClick={() => handleClick("7")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                7
              </button>
              <button
                onClick={() => handleClick("8")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                8
              </button>
              <button
                onClick={() => handleClick("9")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                9
              </button>
              <button
                onClick={() => handleClick("*")}
                className="bg-blue-500 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                *
              </button>

              <button
                onClick={() => handleClick("4")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                4
              </button>
              <button
                onClick={() => handleClick("5")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                5
              </button>
              <button
                onClick={() => handleClick("6")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                6
              </button>
              <button
                onClick={() => handleClick("-")}
                className="bg-blue-500 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                -
              </button>

              <button
                onClick={() => handleClick("1")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                1
              </button>
              <button
                onClick={() => handleClick("2")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                2
              </button>
              <button
                onClick={() => handleClick("3")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                3
              </button>
              <button
                onClick={() => handleClick("+")}
                className="bg-blue-500 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                +
              </button>

              <button
                onClick={() => handleClick("0")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg col-span-2 cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                0
              </button>
              <button
                onClick={() => handleClick(".")}
                className="bg-gray-100 p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                .
              </button>
              <button
                onClick={handleResult}
                className="bg-gray-800 text-white p-3 rounded-lg font-bold text-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
              >
                =
              </button>
            </div>

         
            <button
              onClick={() => setIsVisible(false)} 
              className="mt-4 bg-red-500 text-white px-4 py-2 rounded-lg cursor-pointer hover:scale-110 active:scale-100 transition-transform hover:duration-150 active:duration-50"
            >
              Close
            </button>
          </div>
        </Draggable>
      )}
    </>
  );
}

export default Calculator;