using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rhetorica
{
  /// <summary>
  /// public partial class Program
  /// </summary>
  public partial class Program
  {
    // Member variables (fields, constants):

    // Constructors and finalizers:

    // Methods:
    static void Main(string[] args)
    {
      Program p = new Program();
      string response = string.Empty;

      response = p.DoStuff(args);
      Console.WriteLine(response);

      //Miscellaneous.Pause("");
    }
  }
}
