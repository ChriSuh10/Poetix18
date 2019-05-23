// using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

using ExtensionMethods;

using edu.stanford.nlp.parser.lexparser;
using edu.stanford.nlp.ling;
using edu.stanford.nlp.objectbank;
using edu.stanford.nlp.process;
using edu.stanford.nlp.trees;
using edu.stanford.nlp.util;

using LAIR.ResourceAPIs.WordNet;
using LAIR.Collections.Generic;

using opennlp.tools.chunker;
using opennlp.tools.sentdetect;
//using opennlp.tools.tokenize;
using opennlp.tools.namefind;
using opennlp.tools.parser;
using opennlp.tools.postag;

using PorterStemmerAlgorithm;

namespace Rhetorica
{
  [Flags]
  public enum StopWordsOptions
  {
    None = 0x00,
    FirstWord = 0x01,
    LastWord = 0x02,
    MatchTag = 0x04,
    KeepWords = 0x08
  }

  [Flags]
  public enum AnalyzerOptions
  {
    None = 0x00,
    OmitPunctuationTokens = 0x01,
    OmitFalseDuplicatePhrases = 0x02,
    UsePunctuationDelimitedPhrases = 0x04,
    IgnoreMeansInclude = 0x08
  }

  /// <summary>
  /// public class Analyzer
  /// </summary>
  public class Analyzer
  {
    // Member variables (fields, constants):
    // Penn Treebank punctuation: #  $  .  ,  :  (  )  "  ``  ''  `  '
    // N.B. Added ';' and '!' to punctuation list; they aren't among the Penn Treebank symbols, but they can be used to weed out spuriously tagged punctuation (as tokens instead) in text. [11 Mar. 2015]
    public static readonly string PunctuationPatterns = "^(#|\\$|\\.|,|\\:|\\(|-LRB-|\\)|-RRB-|\"|``|''|`|'|-LCB-|-RCB-|-LSB-|-RSB-|;|\\!|\\?)$";
    public static readonly string FloatingPunctuationPatterns = "(#|\\$|\\.|,|\\:|\\(|-LRB-|\\)|-RRB-|\"|``|''|`|'|-LCB-|-RCB-|-LSB-|-RSB-|;|\\!|\\?)";
    public static readonly string StopWords = @"^(i|me|my|myself|we|us|our|ours|ourselves|you|your|yours|yourself|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves|what|which|who|whom|this|that|these|those|am|is|are|was|were|be|been|being|have|has|had|having|do|does|did|doing|would|should|could|ought|'m|'re|'s|'ve|'d|'ll|n't|wo|sha|ca|cannot|a|an|the|and|but|if|or|because|as|until|while|of|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very)$";
    public static readonly string DeterminersConjunctionsPrepositions = @"^(a|an|the|this|that|whose|'s|his|its|my|your|our|her|their|some|any|no|either|neither|every|which|what|and|nor|or|yet|so|abaft|aboard|about|above|absent|across|afore|after|against|along|alongside|amid|amidst|among|amongst|an|anti|anent|anenst|apropos|apud|around|as|aside|astride|at|athwart|atop|ayond|ayont|barring|before|behind|behither|below|beneath|beside|besides|between|betwixen|betwixt|beyond|but|by|chex|circa|c\.|ca\.?|concerning|contra|cum|despite|down|during|ere|except|excluding|failing|following|for|forby|forenenst|fornenst|fornent|from|fromward|froward|frowards|'?gainst|given|in|including|inside|into|lest|like|mid|midst|minus|modulo|near|'?neath|next|nigh|notwithstanding|of|off|on|onto|opposite|out|outside|outwith|over|overthwart|pace|past|per|plus|pro|qua|re|regarding|round|sans|save|since|than|through|thru|throughout|thruout|till|times|to|toward|towards|'?twixt|under|underneath|unlike|until|unto|up|upon|versus|vs\.?|v\.|via|vice|vis-(?:à|a)-vis|with|w\/|within|w\/in|w\/i|without|w\/o|worth)$";
    public static readonly string DeterminersConjunctionsPrepositionsTag = @"^(CC|DT|IN|WDT)$";
    public static readonly string ConjunctionsPrepositionsTag = @"^(CC|IN)$";
    public static readonly string ConjunctionsTag = @"^(CC)$";
    public static readonly string NounsVerbsAdjectivesAdverbsPronounsTagEquivalent = @"(NN|VB|JJ|RB|WP)";
    public static readonly string NounsVerbsAdjectivesAdverbsTag = @"(NN|NNS|NNP|NNPS|VB|VBD|VBG|VBN|VBP|VBZ|JJ|RB)";

    public static readonly string MostCommonPrefixesRe = "^(?<prefix>anti|de|dis|en|em|fore|in|im|il|ir|inter|mid|mis|non|over|pre|re|semi|sub|super|trans|un|under)";
    //public static readonly string[] MostCommonPrefixes = { "anti", "de", "dis", "en", "em", "fore", "in", "im", "il", "ir", "inter", "mid", "mis", "non", "over", "pre", "re", "semi", "sub", "super", "trans", "un", "under" };
    public static readonly List<string> MostCommonPrefixes = new List<string>() { "anti", "de", "dis", "en", "em", "fore", "in", "im", "il", "ir", "inter", "mid", "mis", "non", "over", "pre", "re", "semi", "sub", "super", "trans", "un", "under" };
    public static readonly string NegationPrefixesRe = "^(?<prefix>anti|de|dis|in|im|il|ir|mis|non|un)";
    public static readonly List<string> NegationPrefixes = new List<string>() { "anti", "de", "dis", "in", "im", "il", "ir", "mis", "non", "un" };
    public static readonly string SimilarityPrefixesRe = "^(?<prefix>en|em|fore|inter|mid|over|pre|re|semi|sub|super|trans|under)";
    public static readonly List<string> SimilarityPrefixes = new List<string>() { "en", "em", "fore", "inter", "mid", "over", "pre", "re", "semi", "sub", "super", "trans", "under" };

    public static readonly string MostCommonSuffixesRe = "(?<suffix>able|ible|al|ial|ed|en|er|or|est|ful|ic|ing|ion|tion|ation|ition|ity|ty|ive|ative|itive|less|ly|ment|ness|ous|eous|ious|s|es|y)$";
    public static readonly List<string> MostCommonSuffixes = new List<string>() { "able", "ible", "al", "ial", "ed", "en", "er", "or", "est", "ful", "ic", "ing", "ion", "tion", "ation", "ition", "ity", "ty", "ive", "ative", "itive", "less", "ly", "ment", "ness", "ous", "eous", "ious", "s", "es", "y" };
    // Remove "less" from full suffix list to prevent negation:
    public static readonly string MostCommonSimilaritySuffixesRe = "(?<suffix>able|ible|al|ial|ed|en|er|or|est|ful|ic|ing|ion|tion|ation|ition|ity|ty|ive|ative|itive|ly|ment|ness|ous|eous|ious|s|es|y)$";
    public static readonly List<string> MostCommonSimilaritySuffixes = new List<string>() { "able", "ible", "al", "ial", "ed", "en", "er", "or", "est", "ful", "ic", "ing", "ion", "tion", "ation", "ition", "ity", "ty", "ive", "ative", "itive", "ly", "ment", "ness", "ous", "eous", "ious", "s", "es", "y" };

    protected string _path;
    protected DocumentPreprocessor.DocType _docType;
    protected string _ignore;
    protected string _punctuation;
    protected AnalyzerOptions _options;
    protected Document _document;
    protected List<RhetoricalFigure> _figures = new List<RhetoricalFigure>();

    public class WordNetRelation
    {
      public WordNetEngine.SynSetRelation Relation { get; protected set; }
      public List<string> Words {get; set; }

      public WordNetRelation(WordNetEngine.SynSetRelation relation)
      {
        Relation = relation;
        Words = null;
      }

      public WordNetRelation(WordNetEngine.SynSetRelation relation, List<string> words)
      {
        Relation = relation;
        Words = words;
      }

      public void ClearWords()
      {
        Words.Clear();
      }
    }

    public readonly Miscellaneous.TreeNode<WordNetRelation> WordNetSearchPath;

    // Constructors and finalizers:
    public Analyzer()
    {
      // Hardcoded WordNet search path for oxymoron (could be configurable, but that's probably unnecessary):
      WordNetSearchPath = Miscellaneous.GetPermutationTree<WordNetRelation>(
        // The root node should always be 'WordNetEngine.SynSetRelation.None'.
        new WordNetRelation(WordNetEngine.SynSetRelation.None),
        new List<WordNetRelation>() {
          // More permutation options can be added here, but are probably unnecessary:
          new WordNetRelation(WordNetEngine.SynSetRelation.Antonym),
          new WordNetRelation(WordNetEngine.SynSetRelation.SimilarTo),
          new WordNetRelation(WordNetEngine.SynSetRelation.DerivationallyRelated)
        }, 3); // 'length' should usually be the same as that of the 'List<>' parameter.
    }

    public Analyzer(string path, DocumentPreprocessor.DocType docType = null, string ignore = "", string punctuation = null, AnalyzerOptions options = AnalyzerOptions.None)
      : this()
    {
      _path = path;
      _docType = docType;
      _ignore = ignore;
      _options = options;

      _punctuation = punctuation ?? PunctuationPatterns;

      Open();
    }

    // Enums, Structs, and Classes:

    // Properties:
    public Document Document
    {
      get { return _document; }
    }

    public List<RhetoricalFigure> Figures
    {
      get { return _figures; }
    }

    // Methods:
    private void Load()
    {
      string fileName = Path.GetFileName(_path);
      string rawText = System.IO.File.ReadAllText(_path, Encoding.ASCII);

      //Console.WriteLine("Loading document: " + fileName);
      Console.WriteLine(Environment.NewLine + "Loading document: " + _path + Environment.NewLine);

      _document = new Document(rawText, fileName, ignore: _ignore, options: _options);
      _document.Preprocessor = _docType == null ? new DocumentPreprocessor(_path) : new DocumentPreprocessor(_path, _docType);
    }

    public void Open()
    {
      Load();
    }

    public delegate void FindFiguresCallback(RhetoricalFigures type);

    public void FindRhetoricalFigures(RhetoricalFigures type = RhetoricalFigures.All, int? windowSize = null, object extra = null, FindFiguresCallback callback = null, RhetoricalFigures exclusions = RhetoricalFigures.None)
    {
      bool anadiplosis = (type.HasFlag(RhetoricalFigures.Anadiplosis) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Anadiplosis);
      bool anaphora = (type.HasFlag(RhetoricalFigures.Anaphora) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Anaphora);
      bool antimetabole = (type.HasFlag(RhetoricalFigures.Antimetabole) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Antimetabole);
      bool chiasmus = (type.HasFlag(RhetoricalFigures.Chiasmus) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Chiasmus);
      bool conduplicatio = (type.HasFlag(RhetoricalFigures.Conduplicatio) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Conduplicatio);
      bool epanalepsis = (type.HasFlag(RhetoricalFigures.Epanalepsis) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Epanalepsis);
      bool epistrophe = (type.HasFlag(RhetoricalFigures.Epistrophe) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Epistrophe);
      bool epizeuxis = (type.HasFlag(RhetoricalFigures.Epizeuxis) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Epizeuxis);
      bool isocolon = (type.HasFlag(RhetoricalFigures.Isocolon) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Isocolon);
      bool oxymoron = (type.HasFlag(RhetoricalFigures.Oxymoron) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Oxymoron);
      bool ploce = (type.HasFlag(RhetoricalFigures.Ploce) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Ploce);
      bool polyptoton = (type.HasFlag(RhetoricalFigures.Polyptoton) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Polyptoton);
      bool polysyndeton = (type.HasFlag(RhetoricalFigures.Polysyndeton) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Polysyndeton);
      bool symploce = (type.HasFlag(RhetoricalFigures.Symploce) || type.HasFlag(RhetoricalFigures.All)) && !exclusions.HasFlag(RhetoricalFigures.Symploce);

      if (epizeuxis) {
        if (callback != null) callback(RhetoricalFigures.Epizeuxis);
        RhetoricalFigure.FindEpizeuxis(this, windowSize);
      }
      if (ploce) {
        if (callback != null) callback(RhetoricalFigures.Ploce);
        RhetoricalFigure.FindPloce(this, windowSize);
      }
      if (conduplicatio) {
        if (callback != null) callback(RhetoricalFigures.Conduplicatio);
        RhetoricalFigure.FindConduplicatio(this, windowSize, extra);
      }
      if (polysyndeton) {
        if (callback != null) callback(RhetoricalFigures.Polysyndeton);
        RhetoricalFigure.FindPolysyndeton(this, windowSize, extra);
      }
      if (anaphora) {
        if (callback != null) callback(RhetoricalFigures.Anaphora);
        RhetoricalFigure.FindAnaphora(this, windowSize, extra);
      }
      if (epistrophe) {
        if (callback != null) callback(RhetoricalFigures.Epistrophe);
        RhetoricalFigure.FindEpistrophe(this, windowSize, extra);
      }
      if (symploce) {
        if (callback != null) callback(RhetoricalFigures.Symploce);
        RhetoricalFigure.FindSymploce(this, windowSize, extra);
      }
      if (epanalepsis) {
        if (callback != null) callback(RhetoricalFigures.Epanalepsis);
        RhetoricalFigure.FindEpanalepsis(this, windowSize, extra);
      }
      if (anadiplosis) {
        if (callback != null) callback(RhetoricalFigures.Anadiplosis);
        RhetoricalFigure.FindAnadiplosis(this, windowSize, extra);
      }
      if (antimetabole) {
        if (callback != null) callback(RhetoricalFigures.Antimetabole);
        RhetoricalFigure.FindAntimetabole(this, windowSize, extra);
      }
      if (polyptoton) {
        if (callback != null) callback(RhetoricalFigures.Polyptoton);
        RhetoricalFigure.FindPolyptoton(this, windowSize);
      }
      if (isocolon) {
        if (callback != null) callback(RhetoricalFigures.Isocolon);
        RhetoricalFigure.FindIsocolon(this, windowSize, extra);
      }
      if (chiasmus) {
        if (callback != null) callback(RhetoricalFigures.Chiasmus);
        RhetoricalFigure.FindChiasmus(this, windowSize, extra);
      }
      if (oxymoron) {
        if (callback != null) callback(RhetoricalFigures.Oxymoron);
        RhetoricalFigure.FindOxymoron(this, windowSize, extra);
      }
    }

    // Operators, indexers, and events:
  }

  public static class StanfordDocType
  {
    public readonly static DocumentPreprocessor.DocType Plain = DocumentPreprocessor.DocType.Plain;

    public readonly static DocumentPreprocessor.DocType XML = DocumentPreprocessor.DocType.XML;
  }
}
