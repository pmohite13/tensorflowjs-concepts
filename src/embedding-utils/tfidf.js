const getInverseDocumentFrequency = (documentTokens, documentDictionary) => {
    return documentDictionary.map((token) => 1 + Math.log(documentTokens.length / documentTokens.reduce((acc, curr) => curr.includes(token) ? acc + 1 : acc, 0)))
}

const getTfIdf = (tfs, idfs) => {
    return tfs.map((element, index) => element * idfs[index])
}

const getTermFrequency = (tokens, dictionary) => {
    return dictionary.map((token) => tokens.reduce((acc, curr) => curr.includes(token) ? acc + 1 : acc, 0));
}

export { getInverseDocumentFrequency, getTfIdf, getTermFrequency }