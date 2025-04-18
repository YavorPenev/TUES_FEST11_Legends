import { useEffect, useState } from 'react';
import axios from 'axios';
import Header from './assets/header';
import Footer from './assets/footer';

function CurrencyCalc() {
  const [rates, setRates] = useState(null);
  const [error, setError] = useState(null);
  const [amount, setAmount] = useState(1);
  const [fromCurrency, setFromCurrency] = useState('USD');
  const [toCurrency, setToCurrency] = useState('EUR')
  const [result, setResult] = useState(null);

  useEffect(() => {
    const fetchRates = async () => {
      try {
        const response = await axios.get('https://api.exchangerate-api.com/v4/latest/USD'); // API за валутни курсове
        setRates(response.data.rates);
      } catch (err) {
        setError('Грешка при извличане на валутните курсове.');
      }
    };
    fetchRates();
  }, []);

  const handleConvert = () => {
    if (rates) {
      const fromRate = rates[fromCurrency];
      const toRate = rates[toCurrency];
      const convertedAmount = (amount / fromRate) * toRate;
      setResult(convertedAmount.toFixed(2));
    }
  };

  return (
    <>
      <Header/>
      <div className="p-5 pt-24" style={{ margin: '0 10%' }}>
        <h1 className="text-3xl font-bold mb-6 text-center">Currency Calculator</h1>
        {error && <p className="text-red-500 text-center">{error}</p>}
        {rates ? (
          <>




            <div className="flex flex-wrap items-center justify-center gap-4 mb-6">
              <label className="font-semibold">Amount:</label>
              <input
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                className="p-2 border rounded w-32"
              />
              <label className="font-semibold">From:</label>
              <select
                value={fromCurrency}
                onChange={(e) => setFromCurrency(e.target.value)}
                className="p-2 border rounded"
              >
                {Object.keys(rates).map((currency) => (
                  <option key={currency} value={currency}>
                    {currency}
                  </option>
                ))}
              </select>
              <label className="font-semibold">To:</label>
              <select
                value={toCurrency}
                onChange={(e) => setToCurrency(e.target.value)}
                className="p-2 border rounded"
              >
                {Object.keys(rates).map((currency) => (
                  <option key={currency} value={currency}>
                    {currency}
                  </option>
                ))}
              </select>
              <button
                onClick={handleConvert}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Convert
              </button>
            </div>


            {result && (
              <div className="text-center mb-8">
                <h2 className="text-2xl font-semibold">
                  {amount} {fromCurrency} = {result} {toCurrency}
                </h2>
              </div>
            )}

            <div className="bg-blue-100 p-4 rounded-lg shadow mb-6">
              <p className="text-lg text-gray-700 text-center">
                Easily convert currencies with real-time exchange rates. Simply enter the amount, select the currencies, and click <b>"Convert"</b> to get the latest conversion. Stay updated with accurate rates and make informed financial decisions effortlessly!
              </p>
            </div>

            <h2 className="text-3xl font-bold mt-8 mb-4">Exchange Rates</h2>
            <div className="grid grid-cols-3 gap-4 max-h-96 overflow-y-auto">
              {Object.entries(rates).map(([currency, rate]) => (
                <div key={currency} className="p-3 bg-blue-300 rounded shadow">
                  <p className="font-semibold">{currency}:</p>
                  <p>{rate.toFixed(2)}</p>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p>Loading...</p>
        )}
      </div>
      <Footer/>
    </>
  );
}

export default CurrencyCalc;