using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using ExtensionMethods;

using edu.stanford.nlp.ling;
using edu.stanford.nlp.trees;

using Newtonsoft.Json;

namespace Rhetorica
{
  /// <summary>
  /// public class RhetoricalFigureParameters
  /// </summary>
  public class RhetoricalFigureParameters
  {
    public int? windowSize { get; set; }
    public object extra { get; set; }
  }

  /// <summary>
  /// public partial class Program
  /// </summary>
  public partial class Program
  {
    static void Callback(RhetoricalFigures type)
    {
      var s = string.Empty;
      s += "Finding " + type + "....";

      Console.WriteLine(s);
    }

    protected string DoStuff(string[] args)
    {
      string rv = string.Empty;

      // Put test methods here:
      string[] pathParts = {
        //Repository.LocalTextPath,
        Repository.NlpTextsPath,
        "sonnets.txt"
        //"Washington - Inaugural Address (1789).txt"
        //"Obama - Inaugural Address (2009).txt"
        //"Obama - Inaugural Address (excerpt, 2009).txt"
        //"Churchill - We Shall Fight on the Beaches (1940).txt"
        //"Churchill - We Shall Fight on the Beaches (excerpt, 1940).txt"
        //"Test Sentences.txt"
        //"test.txt"
        //"epizeuxis_test.txt" // and ploce
        //"polysyndeton_test.txt"
        //"anaphora_test.txt" // and epistrophe
        //"epistrophe_test.txt"
        //"symploce_test.txt"
        //"epanalepsis_test.txt"
        //"anadiplosis_test.txt"
        //"antimetabole_test.txt"
        //"polyptoton_test.txt"
        //"isocolon_test.txt"
        //"chiasmus_test.txt"
        //"oxymoron_test.txt"
        //"Stevens - Farewell to Florida.txt"
      };

      var path = Path.Combine(pathParts);

      if (args.Count() > 0) {
        var args0 = args[0].Trim();
        if (args0 != string.Empty) {
          if (File.Exists(args0))
            path = args0;
          else if (File.Exists(Repository.NlpTextsPath + args0))
            path = Repository.NlpTextsPath + args0;
        }
      }

      //var result = Miscellaneous.GetPermutationTree<string>("root", new List<string>() { "antonym", "synonym", "derived" }, 3);

      AnalyzerOptions options = AnalyzerOptions.OmitPunctuationTokens | AnalyzerOptions.OmitFalseDuplicatePhrases | AnalyzerOptions.UsePunctuationDelimitedPhrases;
      string ignore = "";
      Analyzer a = new Analyzer(path, ignore: ignore, options: options);

      TimeSpan begin = Process.GetCurrentProcess().TotalProcessorTime;

      if (args.Count() > 1) { // Deserialize JSON
        var args1 = args[1].Trim();

        var all = false;
        if (args1 == string.Empty)
          args1 = "{ All: {} }";

        var rhetoricalFigureParameters = JsonConvert.DeserializeObject<Dictionary<string, RhetoricalFigureParameters>>(args1);

        RhetoricalFigures exclusions = RhetoricalFigures.None;

        foreach (var rfp in rhetoricalFigureParameters) {
          var key = rfp.Key;
          RhetoricalFigures rhetoricalFigure;
          if (!Enum.TryParse(key, out rhetoricalFigure))
            continue;

          if (rhetoricalFigure == RhetoricalFigures.All) {
            all = true;
            continue;
          }

          var windowSize = rfp.Value.windowSize;
          var extra = rfp.Value.extra;

          exclusions |= rhetoricalFigure;

          a.FindRhetoricalFigures(rhetoricalFigure, windowSize, extra, Callback);
        }

        if (all)
          a.FindRhetoricalFigures(RhetoricalFigures.All, callback: Callback, exclusions: exclusions);
      }
      else {
        //a.FindRhetoricalFigures(RhetoricalFigures.Epizeuxis, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Ploce, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Conduplicatio, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Polysyndeton, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Anaphora, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Epistrophe, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Symploce, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Epanalepsis, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Anadiplosis, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Antimetabole, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Polyptoton, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Isocolon, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Chiasmus, callback: Callback);
        //a.FindRhetoricalFigures(RhetoricalFigures.Oxymoron, callback: Callback);
        a.FindRhetoricalFigures(RhetoricalFigures.All, callback: Callback);
      }

      TimeSpan end = Process.GetCurrentProcess().TotalProcessorTime;

      Console.WriteLine();
      a.Document.WriteLine();
      Console.WriteLine();

      var figureRows = new List<string>();
      var figureColumns = new string[] {
        "figure_id",
        "token_id",
        "type",
        "word",
        "sentence_id",
        "left_edge",
        "right_edge",
        "tag",
        "tag_equiv",
        "depth",
        "stem"
      };

      string sep = ",";
      var header = String.Join(sep, figureColumns);
      figureRows.Add(header);

      int i = 0;
      foreach (var figure in a.Figures) {
        int j = 0;
        foreach (var token in figure.Tokens) {
          if (token.ContainingSentence == null) continue;
          var rowArray = new object[] {
            i, j,
            figure.Type,
            "\"" + token.Word + "\"",
            token.SentenceId, token.Left, token.Right,
            "\"" + token.Tag + "\"",
            "\"" + token.TagEquivalent + "\"",
            token.Depth,
            "\"" + token.Stem + "\""
          };
          var row = String.Join(sep, rowArray);
          figureRows.Add(row);
          j++;
        }
        i++;
      }

      var sentenceRows = new List<string>();
      var sentenceColumns = new string[] {
        "sentence_id",
        "token_id",
        "word",
        "left_edge",
        "right_edge",
        "tag",
        "tag_equiv",
        "depth",
        "stem"
      };

      header = String.Join(sep, sentenceColumns);
      sentenceRows.Add(header);

      i = 0;
      foreach (var sentence in a.Document.Sentences) {
        int j = 0;
        foreach (var token in sentence.Tokens) {
          var rowArray = new object[] {
            i, j,
            "\"" + token.Word + "\"",
            token.Left, token.Right,
            "\"" + token.Tag + "\"",
            "\"" + token.TagEquivalent + "\"",
            token.Depth,
            "\"" + token.Stem + "\""
          };
          var row = String.Join(sep, rowArray);
          sentenceRows.Add(row);
          j++;
        }
        i++;
      }

      figureRows.ForEach(x => Console.WriteLine("{0}", x));
      Console.WriteLine();

      if (args.Count() > 2) { // Write CSV representations of figures to file.
        var args2 = args[2].Trim();
        var args2Csv = args2 + ".csv";
        var args2Doc = args2 + ".doc.csv";

        Console.WriteLine("Writing document: " + args2Csv + Environment.NewLine);
        File.WriteAllLines(args2Csv, figureRows);

        Console.WriteLine("Writing document: " + args2Doc + Environment.NewLine);
        File.WriteAllLines(args2Doc, sentenceRows);
      }

      foreach (var figure in a.Figures)
        Console.WriteLine(figure.Text);
      Console.WriteLine();

      Console.WriteLine("Measured time: " + (end - begin).TotalMilliseconds + " ms; " + (end - begin).TotalSeconds + " s; " + (end - begin).TotalMinutes + "m.");

      return rv;

      // N.B. This returns to method "Main()", in which a console pause may be commented out; uncomment it for testing.
    }
  }
}
