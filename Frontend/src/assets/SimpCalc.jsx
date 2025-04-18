import { useState, useRef } from "react";
import Draggable from "react-draggable"; // Импортиране на Draggable

function Calculator() {
  const [input, setInput] = useState("");
  const nodeRef = useRef(null); // Създаваме референция за Draggable

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
    <Draggable nodeRef={nodeRef}>
      <div
        ref={nodeRef}
        className="fixed bottom-20 right-4 bg-white p-4 rounded-2xl shadow-2xl w-72 border border-gray-200 cursor-move"
      >
        <div className="mb-4">
          <input
            type="text"
            value={input}
            readOnly
            className="w-full p-3 border rounded text-right font-mono text-xl bg-gray-100"
          />
        </div>

        <div className="grid grid-cols-4 gap-2">
          <button
            onClick={handleClear}
            className="bg-red-400 hover:bg-red-500 text-white p-3 rounded-lg font-bold text-lg"
          >
            C
          </button>
          <button
            onClick={handlePercent}
            className="bg-gray-200 hover:bg-gray-300 p-3 rounded-lg font-bold text-lg"
          >
            %
          </button>
          <button
            onClick={handlePower}
            className="bg-gray-200 hover:bg-gray-300 p-3 rounded-lg font-bold text-lg"
          >
            ^
          </button>
          <button
            onClick={() => handleClick("/")}
            className="bg-yellow-400 hover:bg-yellow-500 p-3 rounded-lg font-bold text-lg"
          >
            /
          </button>

          <button
            onClick={() => handleClick("7")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            7
          </button>
          <button
            onClick={() => handleClick("8")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            8
          </button>
          <button
            onClick={() => handleClick("9")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            9
          </button>
          <button
            onClick={() => handleClick("*")}
            className="bg-yellow-400 hover:bg-yellow-500 p-3 rounded-lg font-bold text-lg"
          >
            *
          </button>

          <button
            onClick={() => handleClick("4")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            4
          </button>
          <button
            onClick={() => handleClick("5")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            5
          </button>
          <button
            onClick={() => handleClick("6")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            6
          </button>
          <button
            onClick={() => handleClick("-")}
            className="bg-yellow-400 hover:bg-yellow-500 p-3 rounded-lg font-bold text-lg"
          >
            -
          </button>

          <button
            onClick={() => handleClick("1")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            1
          </button>
          <button
            onClick={() => handleClick("2")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            2
          </button>
          <button
            onClick={() => handleClick("3")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            3
          </button>
          <button
            onClick={() => handleClick("+")}
            className="bg-yellow-400 hover:bg-yellow-500 p-3 rounded-lg font-bold text-lg"
          >
            +
          </button>

          <button
            onClick={() => handleClick("0")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg col-span-2"
          >
            0
          </button>
          <button
            onClick={() => handleClick(".")}
            className="bg-gray-100 hover:bg-gray-200 p-3 rounded-lg font-bold text-lg"
          >
            .
          </button>
          <button
            onClick={handleResult}
            className="bg-green-400 hover:bg-green-500 text-white p-3 rounded-lg font-bold text-lg"
          >
            =
          </button>
        </div>
      </div>
    </Draggable>
  );
}

export default Calculator;
