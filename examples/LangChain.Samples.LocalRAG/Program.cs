using LangChain.Databases.Sqlite;
using LangChain.DocumentLoaders;
using LangChain.Providers.Ollama;
using LangChain.Extensions;
using Ollama;
using LangChain.Chains.CombineDocuments;
using LangChain.Providers;
using static LangChain.Chains.Chain;
using LangChain.Chains.Sequentials;
using LangChain.Abstractions.Chains.Base;
using LangChain.Chains.LLM;
using LangChain.Prompts;
using LangChain.Schema;
using LangChain.Callback;
using LangChain.Base.Tracers;
using LangChain.Base;


var provider = new OllamaProvider();
var embeddingModel = new OllamaEmbeddingModel(provider, id: "all-minilm");
var llm = new OllamaChatModel(provider, id: "llama3");

var vectorDatabase = new SqLiteVectorDatabase(dataSource: "vectors.db");

var vectorCollection = await vectorDatabase.AddDocumentsFromAsync<PdfPigPdfLoader>(
    embeddingModel, // Used to convert text to embeddings
    dimensions: 384, // Should be 384 for all-minilm
    dataSource: DataSource.FromPath("C:\\projects\\study\\AIML\\harry-potter-and-the-philosophers-stone-by-jk-rowling.pdf"),
    collectionName: "harrypotter", // Can be omitted, use if you want to have multiple collections
    textSplitter: new LangChain.Splitters.Text.RecursiveCharacterTextSplitter(["\n\n", "\n", "(?<=\\. )", " ", ""], chunkSize: 384, chunkOverlap: 100),
    loaderSettings: new DocumentLoaderSettings() { ShouldCollectMetadata = true },
    behavior: AddDocumentsToDatabaseBehavior.JustReturnCollectionIfCollectionIsAlreadyExists);


//const string question = "Who are Harry Potter`s the best friends?";
//const string question = "Where Harry Potter live permanently?";
const string question = "What is Harry's Address?";
/*var similarDocuments = await vectorCollection.GetSimilarDocuments(embeddingModel, question, amount: 10, scoreThreshold: 0.999f, searchType: LangChain.Databases.VectorSearchType.MaximumMarginalRelevance);


// Use similar documents and LLM to answer the question
var answer = await llm.GenerateAsync(
    $"""
     Use the following pieces of context to answer the question at the end.
     If the answer is not in context then just say that you don't know, don't try to make up an answer.
     Keep the answer as short as possible.

     {similarDocuments.AsString()}

     Question: {question}
     Helpful Answer:
     """);

Console.WriteLine($"LLM answer: {answer}");

*/

// building a chain
var prompt = """
     Use the following pieces of context to answer the question at the end.
     If the answer is not in context then just say that you don't know, don't try to make up an answer.
     Keep the answer as short as possible.

     {context}

     Question: {question}
     Helpful Answer:
     """;

var text = question;

var llmchain = LLM(llm, outputKey: "result");

var chain =
    Set(text, outputKey: "question")
    | RetrieveDocuments(vectorCollection, embeddingModel, amount: 2, inputKey: "question", outputKey: "docs")
    //| new MapReduceDocumentsChain(input)
    | Do(async (value) =>
    {
        var docs = value["docs"] as List<Document>;
        Console.WriteLine("RAG content:");
        foreach (var doc in docs)
        {
            Console.WriteLine(doc.PageContent);
        }
         Console.WriteLine();

        var map_template =
@"""The following is a set of documents
{input_documents}
Based on this list of docs, please identify the main themes 
Helpful Answer:""";


        var reduce_template = 
@"""Use the following pieces of context to answer the question at the end.
If the answer is not in context then just return empty string without characters, in case you find the answer just copy text form context.
Keep the answer as short as possible.

     Context: 
     {input_documents}

     Question: {question}
     Helpful Answer:
""";
        var reducedLLM = new LlmChain(new LlmChainInput(llm, new PromptTemplate(new PromptTemplateInput(reduce_template.Replace("{question}",question), new[] { "input_documents"}) )) { Verbose = true });

        var reduceDocumentChain = new StuffDocumentsChain(new StuffDocumentsChainInput(reducedLLM));

        var input = new MapReduceDocumentsChainInput
        {
            LlmChain = reducedLLM,
            ReduceDocumentsChain = reduceDocumentChain,
            ReturnIntermediateSteps = true,
            Verbose = true,
            InputKey = "input_documents",
        };

        var mapReduceChain = new MapReduceDocumentsChain(input);
        try
        {
             value["result"] = "Nothing was done";
            /*var result1 = await mapReduceChain.CallAsync(new ChainValues("input_documents", docs!),
            new HandlersCallbacks(
                new List<BaseCallbackHandler>(new []{new ConsoleCallbackHandler()})));
            Console.WriteLine(result1);
            value["docs"] = result1;*/

            var result1 = await mapReduceChain.CombineDocsAsync(docs!, new Dictionary<string, object>(){{"question",question}});
            Console.WriteLine(result1.Output);
            value["result"] = result1.Output;
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
        }
        finally
        {
            Console.WriteLine("Done");
        }
    })
   
    ;
   // | StuffDocuments(inputKey: "docs", outputKey: "context")  // combine documents together and put them into context
   // | Template(prompt)
   // | LLM(llm, outputKey: "result");


/*var retrievChain =
    Set(text, outputKey: "question")
    | RetrieveDocuments(vectorCollection, embeddingModel, amount: 10, inputKey: "question", outputKey: "docs");

var result = await retrievChain.RunAsync<List<Document>>("docs");*/

var result = await chain.RunAsync("result");
Console.WriteLine(result);

/*
var llmchain = new LangChain.Chains.LLM.LlmChain(new LangChain.Chains.LLM.LlmChainInput(llm,));
var input = new MapReduceDocumentsChainInput
{
    LlmChain = llmchain,
    ReduceDocumentsChain = reduceDocumentsChain.Object,
    ReturnIntermediateSteps = true,
    DocumentVariableName = "theme"
};

var chain = new MapReduceDocumentsChain(input);*/