import *  as sw from "stopword";

const stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'];
const tokenize = (text) => {
    const tmpTokens = text.split(/\s+/g);
    // return text ? text.split(/\s+/g) : [];
    return tmpTokens;
}

const removeStopWords = (tokens, language = 'eng') => {
    if (!language) return tokens;
    // if (language in sw) {
    //     return tokens.filter((token) => !sw[language].includes(token) && token.length > 0);
    // }
    // else {
    //     throw Error('Language not supported')
    // }

    return tokens.filter((token) => !stopwords.includes(token) && token.length > 0);
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