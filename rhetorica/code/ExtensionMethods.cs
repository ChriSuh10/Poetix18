using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

using edu.stanford.nlp.ling;
using edu.stanford.nlp.trees;

using Rhetorica;

namespace ExtensionMethods
{
  /// <summary>
  /// public static class StringExtensions
  /// </summary>
  public static class StringExtensions
  {
    public static string RemoveLineBreaks(this string lines)
    {
      return lines.Replace("\r", string.Empty).Replace("\n", string.Empty);
    }

    public static string ReplaceLineBreaks(this string lines, string replacement)
    {
      return lines.Replace("\r\n", replacement).Replace("\r", replacement).Replace("\n", replacement);
    }
  }

  /// <summary>
  /// public static class ObjectExtensions
  /// </summary>
  public static class ObjectExtensions
  {
    /// <summary>
    /// method DeepClone()
    /// See: http://www.code-magazine.com/Article.aspx?quickid=0601121
    /// </summary>
    /// <param name="obj"></param>
    /// <returns></returns>
    public static object DeepClone(this object obj)
    {
      object objResult = null;
      using (MemoryStream ms = new MemoryStream()) {
        BinaryFormatter bf = new BinaryFormatter();
        bf.Serialize(ms, obj);

        ms.Position = 0;
        objResult = bf.Deserialize(ms);
      }

      return objResult;
    }

    /// <summary>
    /// method GetVariableType()
    /// Get compile-time type of variable.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="instance"></param>
    /// <returns></returns>
    public static Type GetVariableType<T>(this T instance)
    {
      return typeof(T);
    }
  }

  /// <summary>
  /// public static class IEnumerableExtensions
  /// </summary>
  public static class IEnumerableExtensions
  {
    public static bool IsSupersetOf<T>(this IEnumerable<T> a, IEnumerable<T> b)
    {
      // Does 'a' contain all items in 'b'? V. http://stackoverflow.com/questions/1520642/does-net-have-a-way-to-check-if-list-a-contains-all-items-in-list-b.
      // Check whether there are any elements in 'b' that aren't in 'a', then invert the results.
      return !b.Except(a).Any();
    }

    public static bool IsProperSupersetOf<T>(this IEnumerable<T> a, IEnumerable<T> b)
    {
      if (a == b)
        return false;
      else
        return a.IsSupersetOf(b);
    }

    public static bool IsSubsetOf<T>(this IEnumerable<T> a, IEnumerable<T> b)
    {
      return b.IsSupersetOf(a);
    }

    public static bool IsProperSubsetOf<T>(this IEnumerable<T> a, IEnumerable<T> b)
    {
      return b.IsProperSupersetOf(a);
    }

    public static bool Overlaps<T>(this IEnumerable<T> a, IEnumerable<T> b)
    {
      return a.IsSubsetOf(b) || a.IsSupersetOf(b);
    }

    public static bool ProperlyOverlaps<T>(this IEnumerable<T> a, IEnumerable<T> b)
    {
      return a.IsProperSubsetOf(b) || a.IsProperSupersetOf(b);
    }
  }

  /// <summary>
  /// public static class ListExtensions
  /// </summary>
  public static class ListExtensions
  {
    // V. http://stackoverflow.com/questions/5592113/c-how-to-split-list-based-on-index
    public static List<List<T>> Split<T>(this List<T> l, int? size = null) 
    {
      int _size = size ?? l.Count / 2;

      var lists = Enumerable.Range(0, (l.Count + _size - 1) / _size)
            .Select(index => l.GetRange(index * _size, Math.Min(_size, l.Count - index * _size)))
            .ToList();

      return lists;
    }
  }

  /// <summary>
  /// public static class ArrayExtensions
  /// </summary>
  public static class ArrayExtensions
  {
    /// <summary>
    /// method Clone()
    /// See: http://stackoverflow.com/questions/222598/how-do-i-clone-a-generic-list-in-c
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="listToClone"></param>
    /// <returns>IList<T></returns>
    public static IList<T> Clone<T>(this IList<T> listToClone) where T : ICloneable
    {
      return listToClone.Select(item => (T)item.Clone()).ToList();
    }

    /// <summary>
    /// method ToStringGeneric()
    /// See: http://www.vcskicks.com/array-to-string.php
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="array"></param>
    /// <param name="delimeter"></param>
    /// <returns>string</returns>
    public static string ToStringGeneric<T>(this IList<T> array, string delimiter)
    {
      string outputString = string.Empty;

      for (int i = 0; i < array.Count; i++) {
        if (array[i] is IList<T>) {
          // Recursively convert nested arrays to strings:
          outputString += ToStringGeneric<T>((IList<T>)array[i], delimiter);
        } else
          outputString += array[i];

        if (i != array.Count - 1)
          outputString += delimiter;
      }

      return outputString;
    }
    // Usage:
    //   myArray.ToStringGeneric(); // Don't need to specify 'T'
    //   myArray.ToStringGeneric<int>();

    /// <summary>
    /// method ContiguousSubsequences()
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="t"></param>
    /// <returns>IList&lt;IList&lt;T&gt;&gt;</returns>
    public static IList<IList<T>> ContiguousSubsequences<T>(this IList<T> t)
    {
      //var subsequences = new List<List<T>>();
      IList<IList<T>> subsequences = new List<IList<T>>();

      for (int i = 0; i < t.Count; ++i) {
        for (int j = i; j < t.Count; ++j) {
          var current = new List<T>(t.Skip(i).Take(t.Count - j).ToArray());
          subsequences.Add(current);
        }
      }

      return subsequences;
    }
  }

  /// <summary>
  /// public static class RandomExtensions
  /// </summary>
  public static class RandomExtensions
  {
    // TODO: implement additional CharType values (e.g. AsciiAny)
    public enum CharType
    {
      AlphabeticLower,
      AlphabeticUpper,
      AlphabeticAny,
      AlphanumericLower,
      AlphanumericUpper,
      AlphanumericAny,
      Numeric
    }

    // 10 digits vs. 52 alphabetic characters (upper & lower);
    // probability of being numeric: 10 / 62 = 0.1612903225806452
    private const double AlphanumericProbabilityNumericAny = 10.0 / 62.0;

    // 10 digits vs. 26 alphabetic characters (upper OR lower);
    // probability of being numeric: 10 / 36 = 0.2777777777777778
    private const double AlphanumericProbabilityNumericCased = 10.0 / 36.0;

    public static bool NextBool(this Random random, double probability)
    {
      return random.NextDouble() <= probability;
    }

    public static bool NextBool(this Random random)
    {
      return random.NextDouble() <= 0.5;
    }

    public static char NextChar(this Random random, CharType mode)
    {
      switch (mode) {
        case CharType.AlphabeticAny:
          return random.NextAlphabeticChar();
        case CharType.AlphabeticLower:
          return random.NextAlphabeticChar(false);
        case CharType.AlphabeticUpper:
          return random.NextAlphabeticChar(true);
        case CharType.AlphanumericAny:
          return random.NextAlphanumericChar();
        case CharType.AlphanumericLower:
          return random.NextAlphanumericChar(false);
        case CharType.AlphanumericUpper:
          return random.NextAlphanumericChar(true);
        case CharType.Numeric:
          return random.NextNumericChar();
        default:
          return random.NextAlphanumericChar();
      }
    }

    public static char NextChar(this Random random)
    {
      return random.NextChar(CharType.AlphanumericAny);
    }

    private static char NextAlphanumericChar(this Random random, bool uppercase)
    {
      bool numeric = random.NextBool(AlphanumericProbabilityNumericCased);

      if (numeric)
        return random.NextNumericChar();
      else
        return random.NextAlphabeticChar(uppercase);
    }

    private static char NextAlphanumericChar(this Random random)
    {
      bool numeric = random.NextBool(AlphanumericProbabilityNumericAny);

      if (numeric)
        return random.NextNumericChar();
      else
        return random.NextAlphabeticChar(random.NextBool());
    }

    private static char NextAlphabeticChar(this Random random, bool uppercase)
    {
      if (uppercase)
        return (char)random.Next(65, 91);
      else
        return (char)random.Next(97, 123);
    }

    private static char NextAlphabeticChar(this Random random)
    {
      return random.NextAlphabeticChar(random.NextBool());
    }

    private static char NextNumericChar(this Random random)
    {
      return (char)random.Next(48, 58);
    }

    public static DateTime NextDateTime(this Random random, DateTime minValue, DateTime maxValue)
    {
      return DateTime.FromOADate(random.NextDouble(minValue.ToOADate(), maxValue.ToOADate()));
    }

    public static DateTime NextDateTime(this Random random)
    {
      return random.NextDateTime(DateTime.MinValue, DateTime.MaxValue);
    }

    public static double NextDouble(this Random random, double minValue, double maxValue)
    {
      if (maxValue < minValue)
        throw new ArgumentException("Minimum value must be less than maximum value.");

      double difference = maxValue - minValue;
      if (!double.IsInfinity(difference))
        return minValue + (random.NextDouble() * difference);

      else {
        // to avoid evaluating to Double.Infinity, we split the range into two halves:
        double halfDifference = (maxValue * 0.5) - (minValue * 0.5);

        // 50/50 chance of returning a value from the first or second half of the range
        if (random.NextBool())
          return minValue + (random.NextDouble() * halfDifference);
        else
          return (minValue + halfDifference) + (random.NextDouble() * halfDifference);
      }
    }

    public static string NextString(this Random random, int numChars, CharType mode)
    {
      char[] chars = new char[numChars];

      for (int i = 0; i < numChars; ++i)
        chars[i] = random.NextChar(mode);

      return new string(chars);
    }

    public static string NextString(this Random random, int numChars)
    {
      return random.NextString(numChars, CharType.AlphanumericAny);
    }

    public static TimeSpan NextTimeSpan(this Random random, TimeSpan minValue, TimeSpan maxValue)
    {
      return TimeSpan.FromMilliseconds(random.NextDouble(minValue.TotalMilliseconds, maxValue.TotalMilliseconds));
    }

    public static TimeSpan NextTimeSpan(this Random random)
    {
      return random.NextTimeSpan(TimeSpan.MinValue, TimeSpan.MaxValue);
    }
  }
}
