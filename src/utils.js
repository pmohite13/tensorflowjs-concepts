import *  as sw from "stopword";


const tokenize = (text) => {
    const tmpTokens = text.split(/\s+/g);
    // return text ? text.split(/\s+/g) : [];
    return tmpTokens;
}

const removeStopWords = (tokens, language = 'eng') => {
    if (!language) return tokens;
    if (language in sw) {
        return tokens.filter((token) => !sw[language].includes(token) && token.length > 0);
    }
    else {
        throw Error('Language not supported')
    }
}

const createDictionaryForTotalTokenCountInAllDocs = (tokens, tempDictionary) => {
    if (tokens && tokens.length > 0) {
        tempDictionary = tokens.reduce((acc, token) => {
            acc[token] = acc[token] === undefined ? 1 : acc[token] += 1
            return acc;
        }, tempDictionary)
    }
    // return tempDictionary;
}



export { tokenize, removeStopWords, createDictionaryForTotalTokenCountInAllDocs }